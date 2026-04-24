//! GPU backend implementation using wgpu (WebGPU).
//!
//! This module provides GPU-accelerated tensor operations using the wgpu crate,
//! which provides a cross-platform abstraction over Vulkan, Metal, DX12, and WebGPU.
//!
//! # Supported Operations
//!
//! The GPU backend supports most common neural network operations:
//! - Element-wise: Add, Sub, Mul, Div, Neg, Abs, Exp, Log, Sqrt
//! - Activations: ReLU, Sigmoid, Tanh, GELU, SiLU, LeakyReLU, Softmax
//! - Matrix operations: MatMul (including batched)
//! - Reductions: Sum, Mean, Variance
//! - Convolutions: Conv2d (stride, padding)
//! - Shape operations: Transpose, Reshape, Broadcast
//!
//! # Example
//!
//! ```ignore
//! use rustyasg::runtime::wgpu_backend::WgpuBackend;
//! use rustyasg::runtime::backend::Backend;
//!
//! // Create GPU backend (async)
//! let backend = pollster::block_on(WgpuBackend::new());
//!
//! // Load data to GPU
//! let device_data = backend.load_data(&data)?;
//!
//! // Run computation graph
//! let (results, memo) = backend.run(&graph, memo)?;
//!
//! // Retrieve results back to CPU
//! let cpu_results = backend.retrieve_data(&results)?;
//! ```
//!
//! # Limitations
//!
//! - Conv2d: groups must be 1, dilation must be (1, 1)
//! - Some operations may have lower precision than CPU due to GPU floating point handling

use super::backend::{Backend, Memo, RuntimeError};
use crate::asg::{Asg, NodeType, Shape, Value};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// Formats a `&[usize]` as a WGSL array initializer literal like `1u, 4u, 2u`.
fn fmt_u32_array(vals: &[usize]) -> String {
    vals.iter()
        .map(|v| format!("{}u", v))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Represents a tensor stored in GPU memory.
///
/// This struct wraps a wgpu buffer and its associated shape information.
/// The buffer contains f32 values in row-major (C-contiguous) order.
#[derive(Debug)]
pub struct GpuTensor {
    /// The underlying GPU buffer containing tensor data
    buffer: wgpu::Buffer,
    /// Shape of the tensor (e.g., [batch, channels, height, width])
    pub shape: Shape,
}

/// GPU backend using wgpu for accelerated tensor computations.
///
/// This backend executes computation graphs on the GPU using WebGPU/Vulkan/Metal/DX12.
/// It compiles WGSL shaders at runtime for each operation type.
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// Error type for GPU initialization failures
#[derive(Debug)]
pub enum GpuInitError {
    /// No suitable GPU adapter found
    NoAdapter,
    /// Failed to create device
    DeviceCreationFailed(wgpu::RequestDeviceError),
}

impl std::fmt::Display for GpuInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuInitError::NoAdapter => write!(f, "No suitable GPU adapter found. Ensure you have a compatible GPU and drivers installed."),
            GpuInitError::DeviceCreationFailed(e) => write!(f, "Failed to create GPU device: {}", e),
        }
    }
}

impl std::error::Error for GpuInitError {}

impl WgpuBackend {
    /// Creates a new GPU backend asynchronously.
    ///
    /// This function requests a GPU adapter and device. It will panic if no
    /// suitable GPU is found. For fallible initialization, use `try_new()`.
    ///
    /// # Panics
    ///
    /// Panics if no GPU adapter is available or device creation fails.
    pub async fn new() -> Self {
        Self::try_new()
            .await
            .expect("Failed to initialize GPU backend")
    }

    /// Attempts to create a new GPU backend, returning an error on failure.
    ///
    /// This is the fallible version of `new()` that allows handling GPU
    /// initialization errors gracefully.
    ///
    /// # Errors
    ///
    /// Returns `GpuInitError::NoAdapter` if no suitable GPU is found.
    /// Returns `GpuInitError::DeviceCreationFailed` if device creation fails.
    pub async fn try_new() -> Result<Self, GpuInitError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or(GpuInitError::NoAdapter)?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .map_err(GpuInitError::DeviceCreationFailed)?;
        Ok(Self { device, queue })
    }

    /// Reads a GPU buffer back to CPU as raw bytes. Used by shaders that fall
    /// back to CPU computation (currently only `Concat`).
    fn read_buffer_bytes(&self, buffer: &wgpu::Buffer) -> Result<Vec<u8>, RuntimeError> {
        let size = buffer.size();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read_buffer_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = sender.send(v);
        });
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver.receive())
            .ok_or_else(|| RuntimeError::MemoryError("read_buffer_bytes: map failed".into()))?
            .map_err(|e| RuntimeError::MemoryError(format!("read_buffer_bytes: {:?}", e)))?;
        let data = slice.get_mapped_range();
        let bytes = data.to_vec();
        drop(data);
        staging.unmap();
        Ok(bytes)
    }

    /// Copies a GPU buffer to a new buffer.
    fn copy_buffer(&self, source: &wgpu::Buffer) -> wgpu::Buffer {
        let size = source.size();
        let dest = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("copy_buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(source, 0, &dest, 0, size);
        self.queue.submit(Some(encoder.finish()));
        dest
    }

    /// Helper to get tensor from memo with proper error handling.
    #[inline]
    fn get_tensor(
        memo: &Memo<GpuTensor>,
        asg_id: crate::asg::AsgId,
        node_id: crate::asg::NodeId,
    ) -> Result<&GpuTensor, RuntimeError> {
        memo.get(&(asg_id, node_id))
            .ok_or(RuntimeError::NodeNotFound(node_id, asg_id))
    }

    /// Dispatches a WGSL shader that runs with `num_workers` global invocations
    /// (one logical "row" or "column" per worker, workgroup size 64).
    ///
    /// This is the preferred helper for shaders where each worker handles one
    /// slice of data independently — e.g. LayerNorm (one row = one worker),
    /// LayerNormGradGamma (one column = one worker), Embedding (one row = one
    /// worker). The worker dimension is completely decoupled from the output
    /// tensor size, so you can iterate over a reduction axis internally.
    fn dispatch_rowwise(
        &self,
        shader_source: &str,
        output_shape: &Shape,
        num_workers: usize,
        inputs: &[&GpuTensor],
    ) -> Result<GpuTensor, RuntimeError> {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("wgpu_rowwise_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("wgpu_rowwise_pipeline"),
                layout: None,
                module: &module,
                entry_point: "main",
            });

        let output_len = output_shape.iter().product::<usize>();
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rowwise_output"),
            size: (output_len.max(1) * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut entries = Vec::with_capacity(inputs.len() + 1);
        for (i, tensor) in inputs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: tensor.buffer.as_entire_binding(),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: inputs.len() as u32,
            resource: output_buffer.as_entire_binding(),
        });

        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rowwise_bind_group"),
            layout: &layout,
            entries: &entries,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let groups = (num_workers as u32).div_ceil(64);
        pass.dispatch_workgroups(groups.max(1), 1, 1);
        drop(pass);
        self.queue.submit(Some(encoder.finish()));

        Ok(GpuTensor {
            buffer: output_buffer,
            shape: output_shape.clone(),
        })
    }

    fn dispatch_shader(
        &self,
        shader_source: &str,
        output_shape: &Shape,
        inputs: &[&GpuTensor],
    ) -> Result<GpuTensor, RuntimeError> {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("wgpu_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("wgpu_pipeline"),
                layout: None,
                module: &module,
                entry_point: "main",
            });

        let output_len = output_shape.iter().product::<usize>();
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_buffer"),
            size: (output_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut bind_group_entries = Vec::new();
        for (i, tensor) in inputs.iter().enumerate() {
            bind_group_entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: tensor.buffer.as_entire_binding(),
            });
        }

        bind_group_entries.push(wgpu::BindGroupEntry {
            binding: inputs.len() as u32,
            resource: output_buffer.as_entire_binding(),
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group"),
            layout: &bind_group_layout,
            entries: &bind_group_entries,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups((output_len as u32).div_ceil(64), 1, 1);
        drop(compute_pass);
        self.queue.submit(Some(encoder.finish()));

        Ok(GpuTensor {
            buffer: output_buffer,
            shape: output_shape.clone(),
        })
    }
}

impl Backend for WgpuBackend {
    type DeviceData = GpuTensor;

    fn load_data(
        &self,
        data: &HashMap<String, Value>,
    ) -> Result<HashMap<String, Self::DeviceData>, RuntimeError> {
        let mut result = HashMap::new();
        for (name, value) in data {
            if let Value::Tensor(tensor) = value {
                let slice = tensor.as_slice().ok_or_else(|| {
                    RuntimeError::MemoryError(format!(
                        "Tensor '{}' is not contiguous in memory",
                        name
                    ))
                })?;
                let bytes = bytemuck::cast_slice(slice);
                let buffer = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(name.as_str()),
                        contents: bytes,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                    });
                result.insert(
                    name.clone(),
                    GpuTensor {
                        buffer,
                        shape: tensor.shape().to_vec(),
                    },
                );
            }
        }
        Ok(result)
    }

    fn run(
        &self,
        main_asg: &Asg,
        mut memo: Memo<Self::DeviceData>,
    ) -> Result<(Vec<Self::DeviceData>, Memo<Self::DeviceData>), RuntimeError> {
        use crate::analysis::shape_inference::ShapeInference;

        let sorted_nodes = ShapeInference::topological_sort(main_asg)
            .map_err(|e| RuntimeError::ShapeError(format!("topological sort failed: {:?}", e)))?;

        for &node_id in &sorted_nodes {
            let node = main_asg
                .get_node(node_id)
                .map_err(|_| RuntimeError::NodeNotFound(node_id, main_asg.id))?;
            if memo.contains_key(&(main_asg.id, node_id)) {
                continue;
            }

            let shape = node.shape.as_ref().ok_or_else(|| {
                RuntimeError::ShapeError(format!("missing shape for node {}", node_id))
            })?;

            let output = match &node.node_type {
                NodeType::Literal(Value::Tensor(data)) => {
                    let slice = data.as_slice().ok_or_else(|| {
                        RuntimeError::MemoryError("Literal tensor is not contiguous".to_string())
                    })?;
                    let bytes = bytemuck::cast_slice(slice);
                    let buffer =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("literal"),
                                contents: bytes,
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_SRC
                                    | wgpu::BufferUsages::COPY_DST,
                            });
                    GpuTensor {
                        buffer,
                        shape: shape.clone(),
                    }
                }

                NodeType::Add(a, b)
                | NodeType::Subtract(a, b)
                | NodeType::Multiply(a, b)
                | NodeType::Divide(a, b)
                | NodeType::GreaterThan(a, b) => {
                    let lhs = Self::get_tensor(&memo, main_asg.id, *a)?;
                    let rhs = Self::get_tensor(&memo, main_asg.id, *b)?;

                    let op_char = match &node.node_type {
                        NodeType::Add(_, _) => "+",
                        NodeType::Subtract(_, _) => "-",
                        NodeType::Multiply(_, _) => "*",
                        NodeType::Divide(_, _) => "/",
                        NodeType::GreaterThan(_, _) => ">",
                        _ => unreachable!(),
                    };

                    let (result_expr, result_type) =
                        if let NodeType::GreaterThan(_, _) = &node.node_type {
                            ("select(0.0, 1.0, op_result)", "f32")
                        } else {
                            ("op_result", "f32")
                        };

                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> a: array<f32>;
                        @group(0) @binding(1) var<storage, read> b: array<f32>;
                        @group(0) @binding(2) var<storage, read_write> o: array<{result_type}>;

                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let idx = id.x;
                            if (idx >= arrayLength(&o)) {{ return; }}

                            let a_len = arrayLength(&a);
                            let b_len = arrayLength(&b);

                            let a_val = a[select(idx, 0u, a_len == 1u)];
                            let b_val = b[select(idx, 0u, b_len == 1u)];

                            let op_result = a_val {op_char} b_val;
                            o[idx] = {result_expr};
                        }}
                        "#,
                        op_char = op_char,
                        result_expr = result_expr,
                        result_type = result_type
                    );
                    self.dispatch_shader(&shader, shape, &[lhs, rhs])?
                }

                NodeType::Power(base_id, exp_id) => {
                    let base = Self::get_tensor(&memo, main_asg.id, *base_id)?;
                    let exp_node_id = *exp_id;

                    // --- BEGIN FIX ---
                    // Check whether the exponent is the constant 2.0.
                    let is_square_op = main_asg
                        .get_node(exp_node_id)
                        .map(|node| match &node.node_type {
                            // FIXED: the proper way to check for a scalar literal with a specific value.
                            NodeType::Literal(Value::Tensor(t)) => {
                                t.ndim() == 0 && t.iter().next().is_some_and(|&v| v == 2.0)
                            }
                            _ => false,
                        })
                        .unwrap_or(false);

                    let shader = if is_square_op {
                        // For squaring, use the numerically stable multiplication path.
                        r#"
                        @group(0) @binding(0) var<storage, read> base: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> out: array<f32>;
                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let idx = id.x;
                            if (idx >= arrayLength(&out)) { return; }
                            let val = base[idx];
                            out[idx] = val * val;
                        }
                        "#
                        .to_string()
                    } else {
                        // Otherwise, use the generic pow function.
                        r#"
                        @group(0) @binding(0) var<storage, read> base: array<f32>;
                        @group(0) @binding(1) var<storage, read> exp: array<f32>;
                        @group(0) @binding(2) var<storage, read_write> out: array<f32>;
                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let idx = id.x;
                            if (idx >= arrayLength(&out)) { return; }
                            let exp_val = exp[0];
                            out[idx] = pow(base[idx], exp_val);
                        }
                        "#
                        .to_string()
                    };

                    if is_square_op {
                        self.dispatch_shader(&shader, shape, &[base])?
                    } else {
                        let exp = Self::get_tensor(&memo, main_asg.id, exp_node_id)?;
                        self.dispatch_shader(&shader, shape, &[base, exp])?
                    }
                    // --- END FIX ---
                }

                NodeType::MatrixMultiply(a, b) => {
                    let lhs = Self::get_tensor(&memo, main_asg.id, *a)?;
                    let rhs = Self::get_tensor(&memo, main_asg.id, *b)?;

                    let m = lhs.shape[lhs.shape.len() - 2];
                    let k = lhs.shape[lhs.shape.len() - 1];
                    let n = rhs.shape[rhs.shape.len() - 1];

                    let batch_dims = &lhs.shape[..lhs.shape.len() - 2];
                    let batch_count: usize = batch_dims.iter().product();

                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> a: array<f32>;
                        @group(0) @binding(1) var<storage, read> b: array<f32>;
                        @group(0) @binding(2) var<storage, read_write> o: array<f32>;

                        @compute @workgroup_size(8, 8, 1)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let m: u32 = {m}u;
                            let k: u32 = {k}u;
                            let n: u32 = {n}u;
                            let batch = id.z;
                            let row = id.y;
                            let col = id.x;

                            if (row >= m || col >= n || batch >= {batch_count}u) {{ return; }}

                            let a_offset = batch * m * k;
                            let b_offset = batch * k * n;
                            let o_offset = batch * m * n;

                            var sum = 0.0;
                            for (var i: u32 = 0u; i < k; i = i + 1u) {{
                                let a_idx = a_offset + row * k + i;
                                let b_idx = b_offset + i * n + col;
                                sum = sum + a[a_idx] * b[b_idx];
                            }}

                            let o_idx = o_offset + row * n + col;
                            o[o_idx] = sum;
                        }}
                        "#,
                        m = m,
                        k = k,
                        n = n,
                        batch_count = batch_count
                    );

                    let module = self
                        .device
                        .create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some("batched_matmul_shader"),
                            source: wgpu::ShaderSource::Wgsl(shader.into()),
                        });

                    let pipeline =
                        self.device
                            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                                label: Some("batched_matmul_pipeline"),
                                layout: None,
                                module: &module,
                                entry_point: "main",
                            });

                    let output_len = shape.iter().product::<usize>();
                    let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("output_buffer"),
                        size: (output_len * std::mem::size_of::<f32>()) as u64,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    let bind_group_layout = pipeline.get_bind_group_layout(0);
                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("bind_group"),
                        layout: &bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: lhs.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: rhs.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: output_buffer.as_entire_binding(),
                            },
                        ],
                    });

                    let mut encoder = self
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    compute_pass.set_pipeline(&pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);

                    let workgroup_x = (n as u32).div_ceil(8);
                    let workgroup_y = (m as u32).div_ceil(8);
                    let workgroup_z = batch_count as u32;
                    compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, workgroup_z);

                    drop(compute_pass);
                    self.queue.submit(Some(encoder.finish()));

                    GpuTensor {
                        buffer: output_buffer,
                        shape: shape.clone(),
                    }
                }

                NodeType::ReLU(x) => {
                    let input = Self::get_tensor(&memo, main_asg.id, *x)?;
                    let shader = r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let idx = id.x;
                            if (idx >= arrayLength(&o)) { return; }
                            o[idx] = max(x[idx], 0.0);
                        }
                    "#;
                    self.dispatch_shader(shader, shape, &[input])?
                }

                NodeType::Sigmoid(x) => {
                    let input = Self::get_tensor(&memo, main_asg.id, *x)?;
                    let shader = r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let idx = id.x;
                            if (idx >= arrayLength(&o)) { return; }
                            o[idx] = 1.0 / (1.0 + exp(-x[idx]));
                        }
                    "#;
                    self.dispatch_shader(shader, shape, &[input])?
                }

                NodeType::Tanh(x) => {
                    let input = Self::get_tensor(&memo, main_asg.id, *x)?;
                    let shader = r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let idx = id.x;
                            if (idx >= arrayLength(&o)) { return; }
                            o[idx] = tanh(x[idx]);
                        }
                    "#;
                    self.dispatch_shader(shader, shape, &[input])?
                }

                NodeType::Exp(x) => {
                    let input = Self::get_tensor(&memo, main_asg.id, *x)?;
                    let shader = r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let idx = id.x;
                            if (idx >= arrayLength(&o)) { return; }
                            o[idx] = exp(x[idx]);
                        }
                    "#;
                    self.dispatch_shader(shader, shape, &[input])?
                }

                NodeType::Log(x) => {
                    let input = Self::get_tensor(&memo, main_asg.id, *x)?;
                    let shader = r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let idx = id.x;
                            if (idx >= arrayLength(&o)) { return; }
                            o[idx] = log(x[idx]);
                        }
                    "#;
                    self.dispatch_shader(shader, shape, &[input])?
                }

                NodeType::Neg(x) => {
                    let input = Self::get_tensor(&memo, main_asg.id, *x)?;
                    let shader = r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let idx = id.x;
                            if (idx >= arrayLength(&o)) { return; }
                            o[idx] = -x[idx];
                        }
                    "#;
                    self.dispatch_shader(shader, shape, &[input])?
                }

                NodeType::Abs(x) => {
                    let input = Self::get_tensor(&memo, main_asg.id, *x)?;
                    let shader = r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let idx = id.x;
                            if (idx >= arrayLength(&o)) { return; }
                            o[idx] = abs(x[idx]);
                        }
                    "#;
                    self.dispatch_shader(shader, shape, &[input])?
                }

                NodeType::Clamp(x, min_val, max_val) => {
                    let input = Self::get_tensor(&memo, main_asg.id, *x)?;
                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let idx = id.x;
                            if (idx >= arrayLength(&o)) {{ return; }}
                            o[idx] = clamp(x[idx], {min_val}f, {max_val}f);
                        }}
                    "#,
                        min_val = min_val,
                        max_val = max_val
                    );
                    self.dispatch_shader(&shader, shape, &[input])?
                }

                NodeType::LeakyReLU(x, alpha) => {
                    let input = Self::get_tensor(&memo, main_asg.id, *x)?;
                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let idx = id.x;
                            if (idx >= arrayLength(&o)) {{ return; }}
                            let val = x[idx];
                            o[idx] = select({alpha}f * val, val, val > 0.0);
                        }}
                    "#,
                        alpha = alpha
                    );
                    self.dispatch_shader(&shader, shape, &[input])?
                }

                NodeType::GELU(x) => {
                    let input = Self::get_tensor(&memo, main_asg.id, *x)?;
                    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                    let shader = r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let idx = id.x;
                            if (idx >= arrayLength(&o)) { return; }
                            let val = x[idx];
                            let cdf = 0.5 * (1.0 + tanh(0.7978845608 * (val + 0.044715 * val * val * val)));
                            o[idx] = val * cdf;
                        }
                    "#;
                    self.dispatch_shader(shader, shape, &[input])?
                }

                NodeType::SiLU(x) => {
                    let input = Self::get_tensor(&memo, main_asg.id, *x)?;
                    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
                    let shader = r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let idx = id.x;
                            if (idx >= arrayLength(&o)) { return; }
                            let val = x[idx];
                            o[idx] = val / (1.0 + exp(-val));
                        }
                    "#;
                    self.dispatch_shader(shader, shape, &[input])?
                }

                NodeType::Sqrt(x) => {
                    let input = Self::get_tensor(&memo, main_asg.id, *x)?;
                    let shader = r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let idx = id.x;
                            if (idx >= arrayLength(&o)) { return; }
                            o[idx] = sqrt(x[idx]);
                        }
                    "#;
                    self.dispatch_shader(shader, shape, &[input])?
                }

                NodeType::Softmax(x) => {
                    let input = Self::get_tensor(&memo, main_asg.id, *x)?;
                    let last_dim = input.shape.last().copied().unwrap_or(1);
                    let outer_count = shape.iter().product::<usize>() / last_dim;

                    // Each workgroup processes one row to avoid race conditions
                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
                        @compute @workgroup_size(1)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let outer_idx = id.x;
                            if (outer_idx >= {outer}u) {{ return; }}
                            let offset = outer_idx * {last_dim}u;

                            var max_val: f32 = -3.402823466e+38;
                            for (var i = 0u; i < {last_dim}u; i = i + 1u) {{
                                max_val = max(max_val, x[offset + i]);
                            }}

                            var sum = 0.0;
                            for (var i = 0u; i < {last_dim}u; i = i + 1u) {{
                                let val = exp(x[offset + i] - max_val);
                                o[offset + i] = val;
                                sum = sum + val;
                            }}

                            for (var i = 0u; i < {last_dim}u; i = i + 1u) {{
                                o[offset + i] = o[offset + i] / sum;
                            }}
                        }}
                        "#,
                        outer = outer_count,
                        last_dim = last_dim
                    );

                    // Custom dispatch for softmax - one workgroup per row
                    let module = self
                        .device
                        .create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some("softmax_shader"),
                            source: wgpu::ShaderSource::Wgsl(shader.into()),
                        });

                    let pipeline =
                        self.device
                            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                                label: Some("softmax_pipeline"),
                                layout: None,
                                module: &module,
                                entry_point: "main",
                            });

                    let output_len = shape.iter().product::<usize>();
                    let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("softmax_output"),
                        size: (output_len * std::mem::size_of::<f32>()) as u64,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    let bind_group_layout = pipeline.get_bind_group_layout(0);
                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("softmax_bind_group"),
                        layout: &bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: input.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: output_buffer.as_entire_binding(),
                            },
                        ],
                    });

                    let mut encoder = self
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    compute_pass.set_pipeline(&pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    compute_pass.dispatch_workgroups(outer_count as u32, 1, 1);
                    drop(compute_pass);
                    self.queue.submit(Some(encoder.finish()));

                    GpuTensor {
                        buffer: output_buffer,
                        shape: shape.clone(),
                    }
                }

                NodeType::Sum(x) => {
                    let input = Self::get_tensor(&memo, main_asg.id, *x)?;
                    let len = input.shape.iter().product::<usize>();
                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
                        @compute @workgroup_size(1)
                        fn main() {{
                            var sum = 0.0;
                            for (var i = 0u; i < {len}u; i = i + 1u) {{
                                sum = sum + x[i];
                            }}
                            o[0] = sum;
                        }}
                        "#,
                        len = len
                    );
                    self.dispatch_shader(&shader, shape, &[input])?
                }

                NodeType::Mean(x) => {
                    let input = Self::get_tensor(&memo, main_asg.id, *x)?;
                    let last_dim = input.shape.last().copied().unwrap_or(1);
                    let outer_count = input.shape.iter().rev().skip(1).product::<usize>();

                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let outer_idx = id.x;
                            if (outer_idx >= {outer}u) {{ return; }}
                            let offset = outer_idx * {last_dim}u;
                            var sum = 0.0;
                            for (var i = 0u; i < {last_dim}u; i = i + 1u) {{
                                sum = sum + x[offset + i];
                            }}
                            o[outer_idx] = sum / f32({last_dim}u);
                        }}
                        "#,
                        outer = outer_count,
                        last_dim = last_dim
                    );

                    self.dispatch_shader(&shader, shape, &[input])?
                }

                NodeType::Variance(x) => {
                    let input = Self::get_tensor(&memo, main_asg.id, *x)?;
                    let last_dim = input.shape.last().copied().unwrap_or(1);
                    let outer_count = input.shape.iter().rev().skip(1).product::<usize>();

                    // Numerically stable two-pass variance
                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let outer_idx = id.x;
                            if (outer_idx >= {outer}u) {{ return; }}
                            let offset = outer_idx * {last_dim}u;
                            
                            var sum = 0.0;
                            for (var i = 0u; i < {last_dim}u; i = i + 1u) {{
                                sum = sum + x[offset + i];
                            }}
                            let mean = sum / f32({last_dim}u);
                            
                            var sum_sq_diff = 0.0;
                             for (var i = 0u; i < {last_dim}u; i = i + 1u) {{
                                let diff = x[offset + i] - mean;
                                sum_sq_diff = sum_sq_diff + diff * diff;
                            }}
                            o[outer_idx] = sum_sq_diff / f32({last_dim}u);
                        }}
                        "#,
                        outer = outer_count,
                        last_dim = last_dim
                    );
                    self.dispatch_shader(&shader, shape, &[input])?
                }

                NodeType::Reshape(data_id, _shape_id) => {
                    let data_tensor = Self::get_tensor(&memo, main_asg.id, *data_id)?;
                    GpuTensor {
                        buffer: self.copy_buffer(&data_tensor.buffer),
                        shape: shape.clone(),
                    }
                }

                NodeType::Broadcast(source_id, _target_id) => {
                    let source = Self::get_tensor(&memo, main_asg.id, *source_id)?;
                    let shader = r#"
                        @group(0) @binding(0) var<storage, read> src: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> dst: array<f32>;
                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let idx = id.x;
                            if (idx >= arrayLength(&dst)) { return; }
                            dst[idx] = src[0];
                        }
                    "#;
                    self.dispatch_shader(shader, shape, &[source])?
                }

                NodeType::Transpose(data_id, ax1, ax2) => {
                    let input = Self::get_tensor(&memo, main_asg.id, *data_id)?;
                    let rank = input.shape.len();
                    let total_out = shape.iter().product::<usize>();

                    let mut strides_out = vec![0; rank];
                    strides_out[rank - 1] = 1;
                    for i in (0..rank - 1).rev() {
                        strides_out[i] = strides_out[i + 1] * shape[i + 1];
                    }

                    let mut strides_in = vec![0; rank];
                    strides_in[rank - 1] = 1;
                    for i in (0..rank - 1).rev() {
                        strides_in[i] = strides_in[i + 1] * input.shape[i + 1];
                    }

                    let coord_code = (0..rank).map(|i| {
                        format!("let c{i} = rem / {s}u; coords[{i}] = c{i}; rem = rem - c{i} * {s}u;", s = strides_out[i])
                    }).collect::<Vec<_>>().join("\n                            ");

                    let index_code = (0..rank)
                        .map(|i| format!("idx = idx + in_coords[{i}] * {s}u;", s = strides_in[i]))
                        .collect::<Vec<_>>()
                        .join("\n                            ");

                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> src: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> dst: array<f32>;

                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let out_idx = id.x;
                            if (out_idx >= {total_out}u) {{ return; }}

                            var coords: array<u32, {rank}>;
                            var rem = out_idx;
                            {coord_code}

                            var in_coords = coords;
                            let tmp = in_coords[{ax1}];
                            in_coords[{ax1}] = in_coords[{ax2}];
                            in_coords[{ax2}] = tmp;
                            
                            var idx = 0u;
                            {index_code}
                            
                            dst[out_idx] = src[idx];
                        }}
                        "#,
                        rank = rank,
                        ax1 = ax1,
                        ax2 = ax2,
                        total_out = total_out,
                        coord_code = coord_code,
                        index_code = index_code
                    );
                    self.dispatch_shader(&shader, shape, &[input])?
                }

                NodeType::External {
                    source_asg_id,
                    source_node_id,
                    ..
                } => {
                    let tensor = memo
                        .get(&(*source_asg_id, *source_node_id))
                        .ok_or(RuntimeError::NodeNotFound(*source_node_id, *source_asg_id))?;
                    GpuTensor {
                        buffer: self.copy_buffer(&tensor.buffer),
                        shape: tensor.shape.clone(),
                    }
                }

                // Input and Parameter nodes are already in memo from load_data
                NodeType::Input { .. } | NodeType::Parameter { .. } => {
                    continue;
                }

                // ReduceSumTo - reduces gradient to match parameter shape
                NodeType::ReduceSumTo(grad_id, target_id) => {
                    let grad = Self::get_tensor(&memo, main_asg.id, *grad_id)?;
                    let target = Self::get_tensor(&memo, main_asg.id, *target_id)?;
                    let grad_shape = &grad.shape;
                    let target_shape = &target.shape;
                    let out_size: usize = target_shape.iter().product();
                    let in_size: usize = grad_shape.iter().product();

                    if in_size == out_size && grad_shape == target_shape {
                        // Shapes match - just copy
                        GpuTensor {
                            buffer: self.copy_buffer(&grad.buffer),
                            shape: target_shape.clone(),
                        }
                    } else {
                        // Need to reduce by summing
                        let repeat_factor = in_size / out_size.max(1);
                        let shader = format!(
                            r#"
                            @group(0) @binding(0) var<storage, read> grad: array<f32>;
                            @group(0) @binding(1) var<storage, read_write> out: array<f32>;
                            @compute @workgroup_size(64)
                            fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                                let out_idx = id.x;
                                if (out_idx >= {out_size}u) {{ return; }}
                                var sum = 0.0;
                                for (var i = 0u; i < {repeat}u; i = i + 1u) {{
                                    sum = sum + grad[out_idx + i * {out_size}u];
                                }}
                                out[out_idx] = sum;
                            }}
                        "#,
                            out_size = out_size,
                            repeat = repeat_factor
                        );
                        self.dispatch_shader(&shader, target_shape, &[grad])?
                    }
                }

                // Conv2d: Direct convolution implementation
                NodeType::Conv2d {
                    input,
                    weight,
                    bias,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    if *groups != 1 {
                        return Err(RuntimeError::UnimplementedOperation(
                            "Conv2d with groups != 1 not yet supported on GPU".to_string(),
                        ));
                    }
                    if *dilation != (1, 1) {
                        return Err(RuntimeError::UnimplementedOperation(
                            "Conv2d with dilation != (1,1) not yet supported on GPU".to_string(),
                        ));
                    }

                    let input_tensor = Self::get_tensor(&memo, main_asg.id, *input)?;
                    let weight_tensor = Self::get_tensor(&memo, main_asg.id, *weight)?;

                    // Input: [N, C_in, H_in, W_in]
                    // Weight: [C_out, C_in, kH, kW]
                    // Output: [N, C_out, H_out, W_out]
                    let batch_size = input_tensor.shape[0];
                    let in_channels = input_tensor.shape[1];
                    let in_h = input_tensor.shape[2];
                    let in_w = input_tensor.shape[3];
                    let out_channels = weight_tensor.shape[0];
                    let kernel_h = weight_tensor.shape[2];
                    let kernel_w = weight_tensor.shape[3];
                    let (stride_h, stride_w) = stride;
                    let (pad_h, pad_w) = padding;
                    let out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
                    let out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

                    let output_size = batch_size * out_channels * out_h * out_w;

                    // Build WGSL shader for direct convolution
                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> input: array<f32>;
                        @group(0) @binding(1) var<storage, read> weight: array<f32>;
                        @group(0) @binding(2) var<storage, read_write> output: array<f32>;

                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let out_idx = id.x;
                            if (out_idx >= {output_size}u) {{ return; }}

                            // Decode output index: [n, oc, oh, ow]
                            let N = {batch_size}u;
                            let C_out = {out_channels}u;
                            let H_out = {out_h}u;
                            let W_out = {out_w}u;
                            let C_in = {in_channels}u;
                            let H_in = {in_h}u;
                            let W_in = {in_w}u;
                            let kH = {kernel_h}u;
                            let kW = {kernel_w}u;
                            let stride_h = {stride_h}u;
                            let stride_w = {stride_w}u;
                            let pad_h = {pad_h}u;
                            let pad_w = {pad_w}u;

                            let ow = out_idx % W_out;
                            let oh = (out_idx / W_out) % H_out;
                            let oc = (out_idx / (W_out * H_out)) % C_out;
                            let n = out_idx / (W_out * H_out * C_out);

                            var sum: f32 = 0.0;

                            for (var ic = 0u; ic < C_in; ic = ic + 1u) {{
                                for (var kh = 0u; kh < kH; kh = kh + 1u) {{
                                    for (var kw = 0u; kw < kW; kw = kw + 1u) {{
                                        let ih_s = i32(oh * stride_h + kh) - i32(pad_h);
                                        let iw_s = i32(ow * stride_w + kw) - i32(pad_w);

                                        if (ih_s >= 0 && ih_s < i32(H_in) && iw_s >= 0 && iw_s < i32(W_in)) {{
                                            let ih = u32(ih_s);
                                            let iw = u32(iw_s);
                                            let input_idx = n * (C_in * H_in * W_in) + ic * (H_in * W_in) + ih * W_in + iw;
                                            let weight_idx = oc * (C_in * kH * kW) + ic * (kH * kW) + kh * kW + kw;
                                            sum = sum + input[input_idx] * weight[weight_idx];
                                        }}
                                    }}
                                }}
                            }}

                            output[out_idx] = sum;
                        }}
                        "#,
                        output_size = output_size,
                        batch_size = batch_size,
                        out_channels = out_channels,
                        out_h = out_h,
                        out_w = out_w,
                        in_channels = in_channels,
                        in_h = in_h,
                        in_w = in_w,
                        kernel_h = kernel_h,
                        kernel_w = kernel_w,
                        stride_h = stride_h,
                        stride_w = stride_w,
                        pad_h = pad_h,
                        pad_w = pad_w,
                    );

                    let module = self
                        .device
                        .create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some("conv2d_shader"),
                            source: wgpu::ShaderSource::Wgsl(shader.into()),
                        });

                    let pipeline =
                        self.device
                            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                                label: Some("conv2d_pipeline"),
                                layout: None,
                                module: &module,
                                entry_point: "main",
                            });

                    let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("conv2d_output"),
                        size: (output_size * std::mem::size_of::<f32>()) as u64,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    let bind_group_layout = pipeline.get_bind_group_layout(0);
                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("conv2d_bind_group"),
                        layout: &bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: input_tensor.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: weight_tensor.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: output_buffer.as_entire_binding(),
                            },
                        ],
                    });

                    let mut encoder = self
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    compute_pass.set_pipeline(&pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    compute_pass.dispatch_workgroups((output_size as u32).div_ceil(64), 1, 1);
                    drop(compute_pass);
                    self.queue.submit(Some(encoder.finish()));

                    // Handle bias if present
                    let result = if let Some(bias_id) = bias {
                        let bias_tensor = Self::get_tensor(&memo, main_asg.id, *bias_id)?;

                        // Add bias: output[n, oc, oh, ow] += bias[oc]
                        let bias_shader = format!(
                            r#"
                            @group(0) @binding(0) var<storage, read> conv_output: array<f32>;
                            @group(0) @binding(1) var<storage, read> bias: array<f32>;
                            @group(0) @binding(2) var<storage, read_write> output: array<f32>;

                            @compute @workgroup_size(64)
                            fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                                let out_idx = id.x;
                                if (out_idx >= {output_size}u) {{ return; }}

                                let C_out = {out_channels}u;
                                let H_out = {out_h}u;
                                let W_out = {out_w}u;

                                let oc = (out_idx / (W_out * H_out)) % C_out;
                                output[out_idx] = conv_output[out_idx] + bias[oc];
                            }}
                            "#,
                            output_size = output_size,
                            out_channels = out_channels,
                            out_h = out_h,
                            out_w = out_w,
                        );

                        let bias_module =
                            self.device
                                .create_shader_module(wgpu::ShaderModuleDescriptor {
                                    label: Some("conv2d_bias_shader"),
                                    source: wgpu::ShaderSource::Wgsl(bias_shader.into()),
                                });

                        let bias_pipeline =
                            self.device
                                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                                    label: Some("conv2d_bias_pipeline"),
                                    layout: None,
                                    module: &bias_module,
                                    entry_point: "main",
                                });

                        let final_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("conv2d_final_output"),
                            size: (output_size * std::mem::size_of::<f32>()) as u64,
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_SRC
                                | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });

                        let bias_bind_group_layout = bias_pipeline.get_bind_group_layout(0);
                        let bias_bind_group =
                            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("conv2d_bias_bind_group"),
                                layout: &bias_bind_group_layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: output_buffer.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: bias_tensor.buffer.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: final_buffer.as_entire_binding(),
                                    },
                                ],
                            });

                        let mut encoder = self
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                        let mut compute_pass =
                            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                        compute_pass.set_pipeline(&bias_pipeline);
                        compute_pass.set_bind_group(0, &bias_bind_group, &[]);
                        compute_pass.dispatch_workgroups((output_size as u32).div_ceil(64), 1, 1);
                        drop(compute_pass);
                        self.queue.submit(Some(encoder.finish()));

                        GpuTensor {
                            buffer: final_buffer,
                            shape: shape.clone(),
                        }
                    } else {
                        GpuTensor {
                            buffer: output_buffer,
                            shape: shape.clone(),
                        }
                    };

                    result
                }

                NodeType::Slice {
                    input,
                    axis,
                    start,
                    end,
                } => {
                    let x_t = Self::get_tensor(&memo, main_asg.id, *input)?;
                    let in_shape = &x_t.shape;
                    let rank = in_shape.len();
                    let slice_len = end - start;

                    // Compute strides: product of dims to the right.
                    let mut in_strides = vec![1usize; rank];
                    for i in (0..rank.saturating_sub(1)).rev() {
                        in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
                    }
                    let out_shape_vec = shape.clone();
                    let mut out_strides = vec![1usize; rank];
                    for i in (0..rank.saturating_sub(1)).rev() {
                        out_strides[i] = out_strides[i + 1] * out_shape_vec[i + 1];
                    }
                    let output_size: usize = out_shape_vec.iter().product();
                    let axis_start = *start;
                    let axis_val = *axis;
                    let _ = slice_len;

                    // Flatten index -> multi-index -> offset axis by start -> flat input index.
                    // For minimal WGSL complexity: unroll per-axis offsets using constants.
                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> y: array<f32>;

                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let t = id.x;
                            if (t >= {output_size}u) {{ return; }}
                            let rank = {rank}u;
                            var out_strides = array<u32, {rank}>({out_strides_init});
                            var in_strides  = array<u32, {rank}>({in_strides_init});
                            let axis = {axis}u;
                            let axis_start = {axis_start}u;

                            // Decode t -> per-axis index in output, add start on axis, encode to input flat.
                            var rem = t;
                            var in_flat: u32 = 0u;
                            for (var k: u32 = 0u; k < rank; k = k + 1u) {{
                                let idx_k = rem / out_strides[k];
                                rem = rem - idx_k * out_strides[k];
                                let in_idx = select(idx_k, idx_k + axis_start, k == axis);
                                in_flat = in_flat + in_idx * in_strides[k];
                            }}
                            y[t] = x[in_flat];
                        }}
                        "#,
                        output_size = output_size,
                        rank = rank,
                        axis = axis_val,
                        axis_start = axis_start,
                        in_strides_init = fmt_u32_array(&in_strides),
                        out_strides_init = fmt_u32_array(&out_strides),
                    );

                    self.dispatch_shader(&shader, shape, &[x_t])?
                }

                NodeType::Concat { inputs, axis } => {
                    // CPU-side concat for now: ship each input back to CPU, stack, re-upload.
                    // Slice-heavy RoPE concatenates only a few pairs so the data is tiny, but this
                    // keeps us correct without a complex multi-input WGSL kernel.
                    let mut arrays = Vec::with_capacity(inputs.len());
                    for input_id in inputs {
                        let t = Self::get_tensor(&memo, main_asg.id, *input_id)?;
                        let bytes = self.read_buffer_bytes(&t.buffer)?;
                        let f32_slice: &[f32] = bytemuck::cast_slice(&bytes);
                        let arr =
                            ndarray::ArrayD::from_shape_vec(t.shape.clone(), f32_slice.to_vec())
                                .map_err(|e| {
                                    RuntimeError::ShapeError(format!("Concat input: {}", e))
                                })?;
                        arrays.push(arr);
                    }
                    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
                    let out = ndarray::concatenate(ndarray::Axis(*axis), &views)
                        .map_err(|e| RuntimeError::ShapeError(format!("Concat: {}", e)))?;
                    let contiguous = out.as_standard_layout().to_owned();
                    let slice = contiguous.as_slice().ok_or_else(|| {
                        RuntimeError::MemoryError("Concat: could not get contiguous slice".into())
                    })?;
                    let bytes: &[u8] = bytemuck::cast_slice(slice);
                    let buffer =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("concat_output"),
                                contents: bytes,
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_SRC
                                    | wgpu::BufferUsages::COPY_DST,
                            });
                    GpuTensor {
                        buffer,
                        shape: shape.clone(),
                    }
                }

                NodeType::SliceBackward {
                    grad_output,
                    axis,
                    start,
                    full_size,
                } => {
                    let g_t = Self::get_tensor(&memo, main_asg.id, *grad_output)?;
                    let g_shape = &g_t.shape;
                    let rank = g_shape.len();
                    let slice_len = g_shape[*axis];

                    let out_shape_vec = shape.clone();
                    let mut out_strides = vec![1usize; rank];
                    for i in (0..rank.saturating_sub(1)).rev() {
                        out_strides[i] = out_strides[i + 1] * out_shape_vec[i + 1];
                    }
                    let mut in_strides = vec![1usize; rank];
                    for i in (0..rank.saturating_sub(1)).rev() {
                        in_strides[i] = in_strides[i + 1] * g_shape[i + 1];
                    }
                    let output_size: usize = out_shape_vec.iter().product();
                    let axis_val = *axis;
                    let axis_start = *start;
                    let axis_end = axis_start + slice_len;
                    let _ = full_size;

                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> g: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> y: array<f32>;

                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let t = id.x;
                            if (t >= {output_size}u) {{ return; }}
                            let rank = {rank}u;
                            var out_strides = array<u32, {rank}>({out_strides_init});
                            var in_strides  = array<u32, {rank}>({in_strides_init});
                            let axis = {axis}u;
                            let axis_start = {axis_start}u;
                            let axis_end = {axis_end}u;

                            var rem = t;
                            var within: bool = true;
                            var g_flat: u32 = 0u;
                            for (var k: u32 = 0u; k < rank; k = k + 1u) {{
                                let idx_k = rem / out_strides[k];
                                rem = rem - idx_k * out_strides[k];
                                if (k == axis) {{
                                    if (idx_k < axis_start || idx_k >= axis_end) {{ within = false; }}
                                    else {{ g_flat = g_flat + (idx_k - axis_start) * in_strides[k]; }}
                                }} else {{
                                    g_flat = g_flat + idx_k * in_strides[k];
                                }}
                            }}
                            y[t] = select(0.0, g[g_flat], within);
                        }}
                        "#,
                        output_size = output_size,
                        rank = rank,
                        axis = axis_val,
                        axis_start = axis_start,
                        axis_end = axis_end,
                        in_strides_init = fmt_u32_array(&in_strides),
                        out_strides_init = fmt_u32_array(&out_strides),
                    );

                    self.dispatch_shader(&shader, shape, &[g_t])?
                }

                NodeType::Embedding { indices, weight } => {
                    let idx_t = Self::get_tensor(&memo, main_asg.id, *indices)?;
                    let w_t = Self::get_tensor(&memo, main_asg.id, *weight)?;

                    let embedding_dim = *w_t.shape.last().ok_or_else(|| {
                        RuntimeError::ShapeError("Embedding weight must be 2D".into())
                    })?;
                    let num_indices: usize = idx_t.shape.iter().product();
                    let total = num_indices * embedding_dim;

                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> indices: array<f32>;
                        @group(0) @binding(1) var<storage, read> w: array<f32>;
                        @group(0) @binding(2) var<storage, read_write> y: array<f32>;

                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let t = id.x;
                            if (t >= {total}u) {{ return; }}
                            let D = {embedding_dim}u;
                            let row = t / D;
                            let col = t % D;
                            let idx = u32(indices[row]);
                            y[t] = w[idx * D + col];
                        }}
                        "#,
                        total = total,
                        embedding_dim = embedding_dim,
                    );

                    self.dispatch_shader(&shader, shape, &[idx_t, w_t])?
                }

                NodeType::EmbeddingGrad {
                    grad_output,
                    indices,
                    num_embeddings,
                } => {
                    let grad_t = Self::get_tensor(&memo, main_asg.id, *grad_output)?;
                    let idx_t = Self::get_tensor(&memo, main_asg.id, *indices)?;

                    let embedding_dim = *grad_t.shape.last().ok_or_else(|| {
                        RuntimeError::ShapeError(
                            "EmbeddingGrad grad_output must have >=1 dim".into(),
                        )
                    })?;
                    let num_indices: usize = idx_t.shape.iter().product();
                    let total = *num_embeddings * embedding_dim;

                    // Parallelize over every (vocab_row, embed_col) cell of grad_weight and scan
                    // the flat indices array. No atomics needed: each output cell is written once.
                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> grad_out: array<f32>;
                        @group(0) @binding(1) var<storage, read> indices: array<f32>;
                        @group(0) @binding(2) var<storage, read_write> grad_w: array<f32>;

                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let t = id.x;
                            if (t >= {total}u) {{ return; }}
                            let D = {embedding_dim}u;
                            let N = {num_indices}u;
                            let v = t / D;
                            let d = t % D;
                            var acc: f32 = 0.0;
                            for (var i: u32 = 0u; i < N; i = i + 1u) {{
                                if (u32(indices[i]) == v) {{
                                    acc = acc + grad_out[i * D + d];
                                }}
                            }}
                            grad_w[t] = acc;
                        }}
                        "#,
                        total = total,
                        embedding_dim = embedding_dim,
                        num_indices = num_indices,
                    );

                    self.dispatch_shader(&shader, shape, &[grad_t, idx_t])?
                }

                NodeType::ConvTranspose2d {
                    input,
                    weight,
                    bias,
                    stride,
                    padding,
                    output_padding,
                    dilation,
                    groups,
                } => {
                    if *groups != 1 {
                        return Err(RuntimeError::UnimplementedOperation(
                            "ConvTranspose2d with groups != 1 not yet supported on GPU".into(),
                        ));
                    }

                    let x_t = Self::get_tensor(&memo, main_asg.id, *input)?;
                    let w_t = Self::get_tensor(&memo, main_asg.id, *weight)?;

                    let n_batch = x_t.shape[0];
                    let in_channels = x_t.shape[1];
                    let in_h = x_t.shape[2];
                    let in_w = x_t.shape[3];
                    let out_channels_per_group = w_t.shape[1];
                    let kernel_h = w_t.shape[2];
                    let kernel_w = w_t.shape[3];

                    let (stride_h, stride_w) = *stride;
                    let (pad_h, pad_w) = *padding;
                    let (out_pad_h, out_pad_w) = *output_padding;
                    let (dil_h, dil_w) = *dilation;

                    let out_h =
                        (in_h - 1) * stride_h + dil_h * (kernel_h - 1) + out_pad_h + 1 - 2 * pad_h;
                    let out_w =
                        (in_w - 1) * stride_w + dil_w * (kernel_w - 1) + out_pad_w + 1 - 2 * pad_w;

                    let output_size = n_batch * out_channels_per_group * out_h * out_w;

                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read> w: array<f32>;
                        @group(0) @binding(2) var<storage, read_write> y: array<f32>;

                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let idx = id.x;
                            if (idx >= {output_size}u) {{ return; }}
                            let C_in = {in_channels}u;
                            let H_in = {in_h}u;
                            let W_in = {in_w}u;
                            let C_out = {out_channels}u;
                            let H_out = {out_h}u;
                            let W_out = {out_w}u;
                            let kH = {kernel_h}u;
                            let kW = {kernel_w}u;
                            let sh = {stride_h}i;
                            let sw = {stride_w}i;
                            let dh = {dil_h}i;
                            let dw = {dil_w}i;
                            let ph = {pad_h}i;
                            let pw = {pad_w}i;

                            let ow = idx % W_out;
                            let oh = (idx / W_out) % H_out;
                            let oc = (idx / (W_out * H_out)) % C_out;
                            let n  = idx / (W_out * H_out * C_out);

                            var acc: f32 = 0.0;
                            for (var kh: u32 = 0u; kh < kH; kh = kh + 1u) {{
                                let oh_num = i32(oh) + ph - i32(kh) * dh;
                                if (oh_num < 0) {{ continue; }}
                                if ((oh_num % sh) != 0) {{ continue; }}
                                let ih = oh_num / sh;
                                if (ih < 0 || ih >= i32(H_in)) {{ continue; }}
                                for (var kw: u32 = 0u; kw < kW; kw = kw + 1u) {{
                                    let ow_num = i32(ow) + pw - i32(kw) * dw;
                                    if (ow_num < 0) {{ continue; }}
                                    if ((ow_num % sw) != 0) {{ continue; }}
                                    let iw = ow_num / sw;
                                    if (iw < 0 || iw >= i32(W_in)) {{ continue; }}
                                    for (var ic: u32 = 0u; ic < C_in; ic = ic + 1u) {{
                                        let x_idx = n * (C_in * H_in * W_in)
                                                  + ic * (H_in * W_in)
                                                  + u32(ih) * W_in
                                                  + u32(iw);
                                        // Weight layout: [C_in, C_out, kH, kW]
                                        let w_idx = ic * (C_out * kH * kW)
                                                  + oc * (kH * kW)
                                                  + kh * kW
                                                  + kw;
                                        acc = acc + x[x_idx] * w[w_idx];
                                    }}
                                }}
                            }}
                            y[idx] = acc;
                        }}
                        "#,
                        output_size = output_size,
                        in_channels = in_channels,
                        in_h = in_h,
                        in_w = in_w,
                        out_channels = out_channels_per_group,
                        out_h = out_h,
                        out_w = out_w,
                        kernel_h = kernel_h,
                        kernel_w = kernel_w,
                        stride_h = stride_h,
                        stride_w = stride_w,
                        dil_h = dil_h,
                        dil_w = dil_w,
                        pad_h = pad_h,
                        pad_w = pad_w,
                    );

                    let main_out = self.dispatch_shader(&shader, shape, &[x_t, w_t])?;

                    if let Some(bias_id) = bias {
                        let b_t = Self::get_tensor(&memo, main_asg.id, *bias_id)?;
                        let bias_shader = format!(
                            r#"
                            @group(0) @binding(0) var<storage, read> conv_out: array<f32>;
                            @group(0) @binding(1) var<storage, read> bias: array<f32>;
                            @group(0) @binding(2) var<storage, read_write> y: array<f32>;

                            @compute @workgroup_size(64)
                            fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                                let idx = id.x;
                                if (idx >= {output_size}u) {{ return; }}
                                let C_out = {out_channels}u;
                                let oc = (idx / ({out_h}u * {out_w}u)) % C_out;
                                y[idx] = conv_out[idx] + bias[oc];
                            }}
                            "#,
                            output_size = output_size,
                            out_channels = out_channels_per_group,
                            out_h = out_h,
                            out_w = out_w,
                        );
                        self.dispatch_shader(&bias_shader, shape, &[&main_out, b_t])?
                    } else {
                        main_out
                    }
                }

                NodeType::MaxPool2d {
                    input,
                    kernel_size,
                    stride,
                } => {
                    let x_t = Self::get_tensor(&memo, main_asg.id, *input)?;
                    let n = x_t.shape[0];
                    let c = x_t.shape[1];
                    let h = x_t.shape[2];
                    let w = x_t.shape[3];
                    let (kh, kw) = *kernel_size;
                    let (sh, sw) = *stride;
                    let out_h = (h - kh) / sh + 1;
                    let out_w = (w - kw) / sw + 1;
                    let output_size = n * c * out_h * out_w;

                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> y: array<f32>;

                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let idx = id.x;
                            if (idx >= {output_size}u) {{ return; }}
                            let H_out = {out_h}u;
                            let W_out = {out_w}u;
                            let C = {c}u;
                            let H = {h}u;
                            let W = {w}u;
                            let kH = {kh}u;
                            let kW = {kw}u;
                            let sh = {sh}u;
                            let sw = {sw}u;

                            let ow = idx % W_out;
                            let oh = (idx / W_out) % H_out;
                            let cc = (idx / (W_out * H_out)) % C;
                            let nn = idx / (W_out * H_out * C);

                            let h_start = oh * sh;
                            let w_start = ow * sw;

                            var best: f32 = -3.4028235e38;
                            for (var i: u32 = 0u; i < kH; i = i + 1u) {{
                                for (var j: u32 = 0u; j < kW; j = j + 1u) {{
                                    let ih = h_start + i;
                                    let iw = w_start + j;
                                    let v = x[nn * (C * H * W) + cc * (H * W) + ih * W + iw];
                                    if (v > best) {{ best = v; }}
                                }}
                            }}
                            y[idx] = best;
                        }}
                        "#,
                        output_size = output_size,
                        out_h = out_h,
                        out_w = out_w,
                        c = c,
                        h = h,
                        w = w,
                        kh = kh,
                        kw = kw,
                        sh = sh,
                        sw = sw,
                    );

                    self.dispatch_shader(&shader, shape, &[x_t])?
                }

                NodeType::MaxUnpool2d {
                    input,
                    original_input,
                    kernel_size,
                    stride,
                } => {
                    let grad_t = Self::get_tensor(&memo, main_asg.id, *input)?;
                    let orig_t = Self::get_tensor(&memo, main_asg.id, *original_input)?;
                    let n = orig_t.shape[0];
                    let c = orig_t.shape[1];
                    let h = orig_t.shape[2];
                    let w = orig_t.shape[3];
                    let out_h = grad_t.shape[2];
                    let out_w = grad_t.shape[3];
                    let (kh, kw) = *kernel_size;
                    let (sh, sw) = *stride;
                    let input_size = n * c * h * w;

                    // Parallelize over every position of the original input. For each (n,c,ih,iw)
                    // we find every pooling window that covers it, re-scan that window to locate
                    // the max, and accumulate grad_out only if our position is the argmax.
                    // This avoids data races without needing atomic f32.
                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> grad_out: array<f32>;
                        @group(0) @binding(1) var<storage, read> orig: array<f32>;
                        @group(0) @binding(2) var<storage, read_write> grad_in: array<f32>;

                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let idx = id.x;
                            if (idx >= {input_size}u) {{ return; }}
                            let C = {c}u;
                            let H = {h}u;
                            let W = {w}u;
                            let H_out = {out_h}u;
                            let W_out = {out_w}u;
                            let kH = {kh}u;
                            let kW = {kw}u;
                            let sh = {sh}u;
                            let sw = {sw}u;

                            let iw = idx % W;
                            let ih = (idx / W) % H;
                            let cc = (idx / (W * H)) % C;
                            let nn = idx / (W * H * C);

                            var acc: f32 = 0.0;

                            for (var kh_i: u32 = 0u; kh_i < kH; kh_i = kh_i + 1u) {{
                                if (ih < kh_i) {{ continue; }}
                                let oh_raw = ih - kh_i;
                                if ((oh_raw % sh) != 0u) {{ continue; }}
                                let oh = oh_raw / sh;
                                if (oh >= H_out) {{ continue; }}

                                for (var kw_i: u32 = 0u; kw_i < kW; kw_i = kw_i + 1u) {{
                                    if (iw < kw_i) {{ continue; }}
                                    let ow_raw = iw - kw_i;
                                    if ((ow_raw % sw) != 0u) {{ continue; }}
                                    let ow = ow_raw / sw;
                                    if (ow >= W_out) {{ continue; }}

                                    // Find argmax in the (oh, ow) window.
                                    let hs = oh * sh;
                                    let ws = ow * sw;
                                    var best: f32 = -3.4028235e38;
                                    var best_r: u32 = 0u;
                                    var best_c: u32 = 0u;
                                    for (var r: u32 = 0u; r < kH; r = r + 1u) {{
                                        for (var cc_k: u32 = 0u; cc_k < kW; cc_k = cc_k + 1u) {{
                                            let v = orig[nn * (C * H * W) + cc * (H * W) + (hs + r) * W + (ws + cc_k)];
                                            if (v > best) {{
                                                best = v;
                                                best_r = r;
                                                best_c = cc_k;
                                            }}
                                        }}
                                    }}
                                    if (best_r == kh_i && best_c == kw_i) {{
                                        let g_idx = nn * (C * H_out * W_out) + cc * (H_out * W_out) + oh * W_out + ow;
                                        acc = acc + grad_out[g_idx];
                                    }}
                                }}
                            }}
                            grad_in[idx] = acc;
                        }}
                        "#,
                        input_size = input_size,
                        c = c,
                        h = h,
                        w = w,
                        out_h = out_h,
                        out_w = out_w,
                        kh = kh,
                        kw = kw,
                        sh = sh,
                        sw = sw,
                    );

                    self.dispatch_shader(&shader, shape, &[grad_t, orig_t])?
                }

                NodeType::AvgPool2d {
                    input,
                    kernel_size,
                    stride,
                    padding,
                } => {
                    let x_t = Self::get_tensor(&memo, main_asg.id, *input)?;
                    let n = x_t.shape[0];
                    let c = x_t.shape[1];
                    let h = x_t.shape[2];
                    let w = x_t.shape[3];
                    let (kh, kw) = *kernel_size;
                    let (sh, sw) = *stride;
                    let (pad_h, pad_w) = *padding;
                    let out_h = (h + 2 * pad_h - kh) / sh + 1;
                    let out_w = (w + 2 * pad_w - kw) / sw + 1;
                    let output_size = n * c * out_h * out_w;

                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> y: array<f32>;

                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let idx = id.x;
                            if (idx >= {output_size}u) {{ return; }}
                            let H_out = {out_h}u;
                            let W_out = {out_w}u;
                            let C = {c}u;
                            let H = {h}u;
                            let W = {w}u;
                            let kH = {kh}u;
                            let kW = {kw}u;
                            let sh = {sh}u;
                            let sw = {sw}u;
                            let ph = {pad_h}i;
                            let pw = {pad_w}i;

                            let ow = idx % W_out;
                            let oh = (idx / W_out) % H_out;
                            let cc = (idx / (W_out * H_out)) % C;
                            let nn = idx / (W_out * H_out * C);

                            var sum: f32 = 0.0;
                            var count: u32 = 0u;
                            for (var i: u32 = 0u; i < kH; i = i + 1u) {{
                                let ih_s = i32(oh * sh + i) - ph;
                                if (ih_s < 0 || ih_s >= i32(H)) {{ continue; }}
                                for (var j: u32 = 0u; j < kW; j = j + 1u) {{
                                    let iw_s = i32(ow * sw + j) - pw;
                                    if (iw_s < 0 || iw_s >= i32(W)) {{ continue; }}
                                    sum = sum + x[nn * (C * H * W) + cc * (H * W) + u32(ih_s) * W + u32(iw_s)];
                                    count = count + 1u;
                                }}
                            }}
                            y[idx] = select(0.0, sum / f32(count), count > 0u);
                        }}
                        "#,
                        output_size = output_size,
                        out_h = out_h,
                        out_w = out_w,
                        c = c,
                        h = h,
                        w = w,
                        kh = kh,
                        kw = kw,
                        sh = sh,
                        sw = sw,
                        pad_h = pad_h,
                        pad_w = pad_w,
                    );

                    self.dispatch_shader(&shader, shape, &[x_t])?
                }

                NodeType::AvgUnpool2d {
                    input,
                    original_input,
                    kernel_size,
                    stride,
                    padding,
                } => {
                    let grad_t = Self::get_tensor(&memo, main_asg.id, *input)?;
                    // Shapes of the original input are needed for the output shape, which is
                    // already resolved by ShapeInference and baked into `shape`. We still
                    // retrieve `original_input` so the memo dependency is preserved (the ASG
                    // pins it), but the tensor data itself isn't used in the shader.
                    let orig_t = Self::get_tensor(&memo, main_asg.id, *original_input)?;
                    let n = orig_t.shape[0];
                    let c = orig_t.shape[1];
                    let h = orig_t.shape[2];
                    let w = orig_t.shape[3];
                    let out_h = grad_t.shape[2];
                    let out_w = grad_t.shape[3];
                    let (kh, kw) = *kernel_size;
                    let (sh, sw) = *stride;
                    let (pad_h, pad_w) = *padding;
                    let input_size = n * c * h * w;

                    // Distribute grad_out[oh,ow] / (kH*kW) to every input position inside the
                    // window (CPU impl divides by the full kernel area, not by valid count).
                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> grad_out: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> grad_in: array<f32>;

                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let idx = id.x;
                            if (idx >= {input_size}u) {{ return; }}
                            let C = {c}u;
                            let H = {h}u;
                            let W = {w}u;
                            let H_out = {out_h}u;
                            let W_out = {out_w}u;
                            let kH = {kh}u;
                            let kW = {kw}u;
                            let sh = {sh}u;
                            let sw = {sw}u;
                            let ph = {pad_h}i;
                            let pw = {pad_w}i;
                            let area = f32(kH * kW);

                            let iw = idx % W;
                            let ih = (idx / W) % H;
                            let cc = (idx / (W * H)) % C;
                            let nn = idx / (W * H * C);

                            // Input position maps to padded coordinate ih_p = ih + pad_h.
                            let ih_p = i32(ih) + ph;
                            let iw_p = i32(iw) + pw;

                            var acc: f32 = 0.0;
                            for (var kh_i: u32 = 0u; kh_i < kH; kh_i = kh_i + 1u) {{
                                let oh_num = ih_p - i32(kh_i);
                                if (oh_num < 0) {{ continue; }}
                                if ((oh_num % i32(sh)) != 0) {{ continue; }}
                                let oh = oh_num / i32(sh);
                                if (oh < 0 || oh >= i32(H_out)) {{ continue; }}
                                for (var kw_i: u32 = 0u; kw_i < kW; kw_i = kw_i + 1u) {{
                                    let ow_num = iw_p - i32(kw_i);
                                    if (ow_num < 0) {{ continue; }}
                                    if ((ow_num % i32(sw)) != 0) {{ continue; }}
                                    let ow = ow_num / i32(sw);
                                    if (ow < 0 || ow >= i32(W_out)) {{ continue; }}
                                    let g_idx = nn * (C * H_out * W_out) + cc * (H_out * W_out) + u32(oh) * W_out + u32(ow);
                                    acc = acc + grad_out[g_idx] / area;
                                }}
                            }}
                            grad_in[idx] = acc;
                        }}
                        "#,
                        input_size = input_size,
                        c = c,
                        h = h,
                        w = w,
                        out_h = out_h,
                        out_w = out_w,
                        kh = kh,
                        kw = kw,
                        sh = sh,
                        sw = sw,
                        pad_h = pad_h,
                        pad_w = pad_w,
                    );

                    self.dispatch_shader(&shader, shape, &[grad_t])?
                }

                NodeType::AdaptiveAvgPool2d { input, output_size } => {
                    let x_t = Self::get_tensor(&memo, main_asg.id, *input)?;
                    let n = x_t.shape[0];
                    let c = x_t.shape[1];
                    let in_h = x_t.shape[2];
                    let in_w = x_t.shape[3];
                    let (out_h, out_w) = *output_size;
                    let total = n * c * out_h * out_w;

                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> y: array<f32>;

                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let idx = id.x;
                            if (idx >= {total}u) {{ return; }}
                            let C = {c}u;
                            let H_in = {in_h}u;
                            let W_in = {in_w}u;
                            let H_out = {out_h}u;
                            let W_out = {out_w}u;

                            let ow = idx % W_out;
                            let oh = (idx / W_out) % H_out;
                            let cc = (idx / (W_out * H_out)) % C;
                            let nn = idx / (W_out * H_out * C);

                            let h_start = (oh * H_in) / H_out;
                            let h_end = ((oh + 1u) * H_in) / H_out;
                            let w_start = (ow * W_in) / W_out;
                            let w_end = ((ow + 1u) * W_in) / W_out;

                            var sum: f32 = 0.0;
                            let count = f32((h_end - h_start) * (w_end - w_start));
                            for (var ih: u32 = h_start; ih < h_end; ih = ih + 1u) {{
                                for (var iw: u32 = w_start; iw < w_end; iw = iw + 1u) {{
                                    sum = sum + x[nn * (C * H_in * W_in) + cc * (H_in * W_in) + ih * W_in + iw];
                                }}
                            }}
                            y[idx] = sum / count;
                        }}
                        "#,
                        total = total,
                        c = c,
                        in_h = in_h,
                        in_w = in_w,
                        out_h = out_h,
                        out_w = out_w,
                    );

                    self.dispatch_shader(&shader, shape, &[x_t])?
                }

                NodeType::Conv2dBackwardInput {
                    grad_output,
                    weight,
                    input_shape,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    if *groups != 1 {
                        return Err(RuntimeError::UnimplementedOperation(
                            "Conv2dBackwardInput with groups != 1 not yet supported on GPU".into(),
                        ));
                    }
                    if *dilation != (1, 1) {
                        return Err(RuntimeError::UnimplementedOperation(
                            "Conv2dBackwardInput with dilation != (1,1) not yet supported on GPU"
                                .into(),
                        ));
                    }

                    let grad_t = Self::get_tensor(&memo, main_asg.id, *grad_output)?;
                    let w_t = Self::get_tensor(&memo, main_asg.id, *weight)?;

                    let (n_batch, in_channels, in_h, in_w) = *input_shape;
                    let out_channels = grad_t.shape[1];
                    let out_h = grad_t.shape[2];
                    let out_w = grad_t.shape[3];
                    let kernel_h = w_t.shape[2];
                    let kernel_w = w_t.shape[3];
                    let (stride_h, stride_w) = *stride;
                    let (pad_h, pad_w) = *padding;

                    let output_size = n_batch * in_channels * in_h * in_w;

                    // Each worker computes one grad_input[n, ic, ih, iw] by iterating over
                    // the (oc, kh, kw) triples that contributed to this input position.
                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> grad_out: array<f32>;
                        @group(0) @binding(1) var<storage, read> weight: array<f32>;
                        @group(0) @binding(2) var<storage, read_write> grad_in: array<f32>;

                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let idx = id.x;
                            if (idx >= {output_size}u) {{ return; }}

                            let C_in = {in_channels}u;
                            let H_in = {in_h}u;
                            let W_in = {in_w}u;
                            let C_out = {out_channels}u;
                            let H_out = {out_h}u;
                            let W_out = {out_w}u;
                            let kH = {kernel_h}u;
                            let kW = {kernel_w}u;
                            let sh = {stride_h}i;
                            let sw = {stride_w}i;
                            let ph = {pad_h}i;
                            let pw = {pad_w}i;

                            let iw = idx % W_in;
                            let ih = (idx / W_in) % H_in;
                            let ic = (idx / (W_in * H_in)) % C_in;
                            let n  = idx / (W_in * H_in * C_in);

                            var acc: f32 = 0.0;

                            // For each kernel position, find matching (oh, ow) such that
                            //   oh*stride + kh - pad == ih  and  ow*stride + kw - pad == iw.
                            for (var kh: u32 = 0u; kh < kH; kh = kh + 1u) {{
                                let oh_num = i32(ih) + ph - i32(kh);
                                if (oh_num < 0) {{ continue; }}
                                if ((oh_num % sh) != 0) {{ continue; }}
                                let oh = oh_num / sh;
                                if (oh < 0 || oh >= i32(H_out)) {{ continue; }}

                                for (var kw: u32 = 0u; kw < kW; kw = kw + 1u) {{
                                    let ow_num = i32(iw) + pw - i32(kw);
                                    if (ow_num < 0) {{ continue; }}
                                    if ((ow_num % sw) != 0) {{ continue; }}
                                    let ow = ow_num / sw;
                                    if (ow < 0 || ow >= i32(W_out)) {{ continue; }}

                                    for (var oc: u32 = 0u; oc < C_out; oc = oc + 1u) {{
                                        let g_idx = n * (C_out * H_out * W_out)
                                                  + oc * (H_out * W_out)
                                                  + u32(oh) * W_out
                                                  + u32(ow);
                                        let w_idx = oc * (C_in * kH * kW)
                                                  + ic * (kH * kW)
                                                  + kh * kW
                                                  + kw;
                                        acc = acc + grad_out[g_idx] * weight[w_idx];
                                    }}
                                }}
                            }}

                            grad_in[idx] = acc;
                        }}
                        "#,
                        output_size = output_size,
                        in_channels = in_channels,
                        in_h = in_h,
                        in_w = in_w,
                        out_channels = out_channels,
                        out_h = out_h,
                        out_w = out_w,
                        kernel_h = kernel_h,
                        kernel_w = kernel_w,
                        stride_h = stride_h,
                        stride_w = stride_w,
                        pad_h = pad_h,
                        pad_w = pad_w,
                    );

                    self.dispatch_shader(&shader, shape, &[grad_t, w_t])?
                }

                NodeType::Conv2dBackwardWeight {
                    grad_output,
                    input,
                    weight_shape,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    if *groups != 1 {
                        return Err(RuntimeError::UnimplementedOperation(
                            "Conv2dBackwardWeight with groups != 1 not yet supported on GPU".into(),
                        ));
                    }
                    if *dilation != (1, 1) {
                        return Err(RuntimeError::UnimplementedOperation(
                            "Conv2dBackwardWeight with dilation != (1,1) not yet supported on GPU"
                                .into(),
                        ));
                    }

                    let grad_t = Self::get_tensor(&memo, main_asg.id, *grad_output)?;
                    let x_t = Self::get_tensor(&memo, main_asg.id, *input)?;

                    let (out_channels, c_in_per_group, kernel_h, kernel_w) = *weight_shape;
                    let n_batch = x_t.shape[0];
                    let in_h = x_t.shape[2];
                    let in_w = x_t.shape[3];
                    let out_h = grad_t.shape[2];
                    let out_w = grad_t.shape[3];
                    let (stride_h, stride_w) = *stride;
                    let (pad_h, pad_w) = *padding;

                    let output_size = out_channels * c_in_per_group * kernel_h * kernel_w;

                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> grad_out: array<f32>;
                        @group(0) @binding(1) var<storage, read> x: array<f32>;
                        @group(0) @binding(2) var<storage, read_write> grad_w: array<f32>;

                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let idx = id.x;
                            if (idx >= {output_size}u) {{ return; }}

                            let C_out = {out_channels}u;
                            let C_in = {c_in}u;
                            let H_in = {in_h}u;
                            let W_in = {in_w}u;
                            let H_out = {out_h}u;
                            let W_out = {out_w}u;
                            let kH = {kernel_h}u;
                            let kW = {kernel_w}u;
                            let N = {n_batch}u;
                            let sh = {stride_h}u;
                            let sw = {stride_w}u;
                            let ph = {pad_h}i;
                            let pw = {pad_w}i;

                            let kw = idx % kW;
                            let kh = (idx / kW) % kH;
                            let ic = (idx / (kW * kH)) % C_in;
                            let oc = idx / (kW * kH * C_in);

                            var acc: f32 = 0.0;
                            for (var n: u32 = 0u; n < N; n = n + 1u) {{
                                for (var oh: u32 = 0u; oh < H_out; oh = oh + 1u) {{
                                    let ih_s = i32(oh * sh) + i32(kh) - ph;
                                    if (ih_s < 0 || ih_s >= i32(H_in)) {{ continue; }}
                                    for (var ow: u32 = 0u; ow < W_out; ow = ow + 1u) {{
                                        let iw_s = i32(ow * sw) + i32(kw) - pw;
                                        if (iw_s < 0 || iw_s >= i32(W_in)) {{ continue; }}
                                        let g_idx = n * (C_out * H_out * W_out)
                                                  + oc * (H_out * W_out)
                                                  + oh * W_out
                                                  + ow;
                                        let x_idx = n * (C_in * H_in * W_in)
                                                  + ic * (H_in * W_in)
                                                  + u32(ih_s) * W_in
                                                  + u32(iw_s);
                                        acc = acc + grad_out[g_idx] * x[x_idx];
                                    }}
                                }}
                            }}
                            grad_w[idx] = acc;
                        }}
                        "#,
                        output_size = output_size,
                        out_channels = out_channels,
                        c_in = c_in_per_group,
                        in_h = in_h,
                        in_w = in_w,
                        out_h = out_h,
                        out_w = out_w,
                        kernel_h = kernel_h,
                        kernel_w = kernel_w,
                        n_batch = n_batch,
                        stride_h = stride_h,
                        stride_w = stride_w,
                        pad_h = pad_h,
                        pad_w = pad_w,
                    );

                    self.dispatch_shader(&shader, shape, &[grad_t, x_t])?
                }

                NodeType::LayerNorm {
                    input,
                    gamma,
                    beta,
                    eps,
                } => {
                    let x_t = Self::get_tensor(&memo, main_asg.id, *input)?;
                    let g_t = Self::get_tensor(&memo, main_asg.id, *gamma)?;
                    let b_t = Self::get_tensor(&memo, main_asg.id, *beta)?;

                    let norm_size = *x_t.shape.last().ok_or_else(|| {
                        RuntimeError::ShapeError(
                            "LayerNorm input must have at least 1 dimension".into(),
                        )
                    })?;
                    let batch_size: usize = x_t.shape.iter().take(x_t.shape.len() - 1).product();

                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read> gamma: array<f32>;
                        @group(0) @binding(2) var<storage, read> beta: array<f32>;
                        @group(0) @binding(3) var<storage, read_write> y: array<f32>;

                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let row = id.x;
                            if (row >= {batch_size}u) {{ return; }}
                            let N = {norm_size}u;
                            let base = row * N;

                            // Mean.
                            var sum: f32 = 0.0;
                            for (var j: u32 = 0u; j < N; j = j + 1u) {{
                                sum = sum + x[base + j];
                            }}
                            let mean = sum / f32(N);

                            // Variance.
                            var vsum: f32 = 0.0;
                            for (var j: u32 = 0u; j < N; j = j + 1u) {{
                                let d = x[base + j] - mean;
                                vsum = vsum + d * d;
                            }}
                            let inv_std = 1.0 / sqrt(vsum / f32(N) + {eps});

                            for (var j: u32 = 0u; j < N; j = j + 1u) {{
                                let xn = (x[base + j] - mean) * inv_std;
                                y[base + j] = xn * gamma[j] + beta[j];
                            }}
                        }}
                        "#,
                        batch_size = batch_size,
                        norm_size = norm_size,
                        eps = eps,
                    );

                    self.dispatch_rowwise(&shader, shape, batch_size, &[x_t, g_t, b_t])?
                }

                NodeType::LayerNormBackward {
                    grad_output,
                    input,
                    gamma,
                    eps,
                } => {
                    let dy_t = Self::get_tensor(&memo, main_asg.id, *grad_output)?;
                    let x_t = Self::get_tensor(&memo, main_asg.id, *input)?;
                    let g_t = Self::get_tensor(&memo, main_asg.id, *gamma)?;

                    let norm_size = *x_t.shape.last().ok_or_else(|| {
                        RuntimeError::ShapeError("LayerNormBackward input must have >=1 dim".into())
                    })?;
                    let batch_size: usize = x_t.shape.iter().take(x_t.shape.len() - 1).product();

                    // dx = inv_std * (dy_gamma - mean(dy_gamma) - x_norm * mean(dy_gamma * x_norm))
                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> dy: array<f32>;
                        @group(0) @binding(1) var<storage, read> x: array<f32>;
                        @group(0) @binding(2) var<storage, read> gamma: array<f32>;
                        @group(0) @binding(3) var<storage, read_write> dx: array<f32>;

                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let row = id.x;
                            if (row >= {batch_size}u) {{ return; }}
                            let N = {norm_size}u;
                            let base = row * N;
                            let n_f = f32(N);

                            var sum: f32 = 0.0;
                            for (var j: u32 = 0u; j < N; j = j + 1u) {{
                                sum = sum + x[base + j];
                            }}
                            let mean = sum / n_f;

                            var vsum: f32 = 0.0;
                            for (var j: u32 = 0u; j < N; j = j + 1u) {{
                                let d = x[base + j] - mean;
                                vsum = vsum + d * d;
                            }}
                            let inv_std = 1.0 / sqrt(vsum / n_f + {eps});

                            var mean_dg: f32 = 0.0;
                            var mean_dg_xn: f32 = 0.0;
                            for (var j: u32 = 0u; j < N; j = j + 1u) {{
                                let dg = dy[base + j] * gamma[j];
                                let xn = (x[base + j] - mean) * inv_std;
                                mean_dg = mean_dg + dg;
                                mean_dg_xn = mean_dg_xn + dg * xn;
                            }}
                            mean_dg = mean_dg / n_f;
                            mean_dg_xn = mean_dg_xn / n_f;

                            for (var j: u32 = 0u; j < N; j = j + 1u) {{
                                let dg = dy[base + j] * gamma[j];
                                let xn = (x[base + j] - mean) * inv_std;
                                dx[base + j] = inv_std * (dg - mean_dg - xn * mean_dg_xn);
                            }}
                        }}
                        "#,
                        batch_size = batch_size,
                        norm_size = norm_size,
                        eps = eps,
                    );

                    self.dispatch_rowwise(&shader, shape, batch_size, &[dy_t, x_t, g_t])?
                }

                NodeType::LayerNormGradGamma {
                    grad_output,
                    input,
                    eps,
                } => {
                    let dy_t = Self::get_tensor(&memo, main_asg.id, *grad_output)?;
                    let x_t = Self::get_tensor(&memo, main_asg.id, *input)?;

                    let norm_size = *x_t.shape.last().ok_or_else(|| {
                        RuntimeError::ShapeError(
                            "LayerNormGradGamma input must have >=1 dim".into(),
                        )
                    })?;
                    let batch_size: usize = x_t.shape.iter().take(x_t.shape.len() - 1).product();

                    // Parallelize over columns j: grad_gamma[j] = sum_row(dy[row,j] * (x[row,j] - mean_row) / std_row).
                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> dy: array<f32>;
                        @group(0) @binding(1) var<storage, read> x: array<f32>;
                        @group(0) @binding(2) var<storage, read_write> grad_gamma: array<f32>;

                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let j = id.x;
                            if (j >= {norm_size}u) {{ return; }}
                            let N = {norm_size}u;
                            let B = {batch_size}u;
                            let n_f = f32(N);

                            var acc: f32 = 0.0;
                            for (var row: u32 = 0u; row < B; row = row + 1u) {{
                                let base = row * N;
                                var sum: f32 = 0.0;
                                for (var k: u32 = 0u; k < N; k = k + 1u) {{
                                    sum = sum + x[base + k];
                                }}
                                let mean = sum / n_f;
                                var vsum: f32 = 0.0;
                                for (var k: u32 = 0u; k < N; k = k + 1u) {{
                                    let d = x[base + k] - mean;
                                    vsum = vsum + d * d;
                                }}
                                let inv_std = 1.0 / sqrt(vsum / n_f + {eps});
                                let x_norm = (x[base + j] - mean) * inv_std;
                                acc = acc + dy[base + j] * x_norm;
                            }}
                            grad_gamma[j] = acc;
                        }}
                        "#,
                        norm_size = norm_size,
                        batch_size = batch_size,
                        eps = eps,
                    );

                    // Dispatch norm_size workers.
                    self.dispatch_rowwise(&shader, shape, norm_size, &[dy_t, x_t])?
                }

                NodeType::LayerNormGradBeta { grad_output } => {
                    let dy_t = Self::get_tensor(&memo, main_asg.id, *grad_output)?;

                    let norm_size = *dy_t.shape.last().ok_or_else(|| {
                        RuntimeError::ShapeError(
                            "LayerNormGradBeta grad_output must have >=1 dim".into(),
                        )
                    })?;
                    let batch_size: usize = dy_t.shape.iter().take(dy_t.shape.len() - 1).product();

                    let shader = format!(
                        r#"
                        @group(0) @binding(0) var<storage, read> dy: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> grad_beta: array<f32>;

                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let j = id.x;
                            if (j >= {norm_size}u) {{ return; }}
                            let N = {norm_size}u;
                            let B = {batch_size}u;
                            var acc: f32 = 0.0;
                            for (var row: u32 = 0u; row < B; row = row + 1u) {{
                                acc = acc + dy[row * N + j];
                            }}
                            grad_beta[j] = acc;
                        }}
                        "#,
                        norm_size = norm_size,
                        batch_size = batch_size,
                    );

                    self.dispatch_rowwise(&shader, shape, norm_size, &[dy_t])?
                }

                _ => {
                    return Err(RuntimeError::UnimplementedOperation(format!(
                        "node type not supported on GPU: {:?}",
                        node.node_type
                    )))
                }
            };

            memo.insert((main_asg.id, node_id), output);
        }

        let mut outputs = Vec::new();
        for &node_id in &main_asg.outputs {
            let tensor = Self::get_tensor(&memo, main_asg.id, node_id)?;
            let buffer = self.copy_buffer(&tensor.buffer);
            outputs.push(GpuTensor {
                buffer,
                shape: tensor.shape.clone(),
            });
        }

        Ok((outputs, memo))
    }

    fn retrieve_data(&self, device_data: &[Self::DeviceData]) -> Result<Vec<Value>, RuntimeError> {
        let mut result = Vec::new();
        for tensor in device_data {
            let buffer_size = tensor.buffer.size();
            let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encoder.copy_buffer_to_buffer(&tensor.buffer, 0, &staging_buffer, 0, buffer_size);
            self.queue.submit(Some(encoder.finish()));

            let buffer_slice = staging_buffer.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
                let _ = sender.send(v);
            });
            self.device.poll(wgpu::Maintain::Wait);

            pollster::block_on(receiver.receive())
                .ok_or_else(|| RuntimeError::MemoryError("GPU buffer map failed".to_string()))?
                .map_err(|e| RuntimeError::MemoryError(format!("GPU buffer map error: {:?}", e)))?;

            let data = buffer_slice.get_mapped_range();
            let slice: &[f32] = bytemuck::cast_slice(&data);
            let array = ndarray::ArrayD::from_shape_vec(tensor.shape.clone(), slice.to_vec())
                .map_err(|e| RuntimeError::ShapeError(format!("Failed to create array: {}", e)))?;
            drop(data);
            staging_buffer.unmap();

            result.push(Value::Tensor(array));
        }
        Ok(result)
    }
}

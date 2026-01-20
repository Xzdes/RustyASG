//! Модуль, реализующий GPU-бэкенд на базе wgpu с корректным WGSL.
//!
//! Все шейдеры переписаны без сокращений и соответствуют спецификации WGSL 1.0.

use super::backend::{Backend, Memo, RuntimeError};
use crate::asg::{Asg, NodeType, Shape, Value};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// Представление тензора в GPU-памяти.
#[derive(Debug)]
pub struct GpuTensor {
    buffer: wgpu::Buffer,
    shape: Shape,
}

/// GPU-бэкенд на базе wgpu.
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl WgpuBackend {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .unwrap();
        Self { device, queue }
    }

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
    let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(source, 0, &dest, 0, size);
    self.queue.submit(Some(encoder.finish()));
    dest
}

    fn dispatch_shader(
        &self,
        shader_source: &str,
        output_shape: &Shape,
        inputs: &[&GpuTensor],
    ) -> Result<GpuTensor, RuntimeError> {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgpu_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("wgpu_pipeline"),
            layout: None,
            module: &module,
            entry_point: "main",
        });

        let output_len = output_shape.iter().product::<usize>();
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_buffer"),
            size: (output_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
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

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(((output_len as u32) + 63) / 64, 1, 1);
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
                let bytes = bytemuck::cast_slice(tensor.as_slice().unwrap());
                let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(name.as_str()),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                });
                result.insert(name.clone(), GpuTensor {
                    buffer,
                    shape: tensor.shape().to_vec(),
                });
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
            let node = main_asg.get_node(node_id).unwrap();
            if memo.contains_key(&(main_asg.id, node_id)) {
                continue;
            }

            let shape = node.shape.as_ref().ok_or_else(|| {
                RuntimeError::ShapeError(format!("missing shape for node {}", node_id))
            })?;

            let output = match &node.node_type {
                NodeType::Literal(Value::Tensor(data)) => {
                    let bytes = bytemuck::cast_slice(data.as_slice().unwrap());
                    let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("literal"),
                        contents: bytes,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    });
                    GpuTensor {
                        buffer,
                        shape: shape.clone(),
                    }
                }
                
                NodeType::Add(a, b) | NodeType::Subtract(a, b) | NodeType::Multiply(a, b) | NodeType::Divide(a, b) | NodeType::GreaterThan(a, b) => {
                    let lhs = memo.get(&(main_asg.id, *a)).unwrap();
                    let rhs = memo.get(&(main_asg.id, *b)).unwrap();

                    let op_char = match &node.node_type {
                        NodeType::Add(_, _) => "+",
                        NodeType::Subtract(_, _) => "-",
                        NodeType::Multiply(_, _) => "*",
                        NodeType::Divide(_, _) => "/",
                        NodeType::GreaterThan(_, _) => ">",
                        _ => unreachable!(),
                    };

                    let (result_expr, result_type) = if let NodeType::GreaterThan(_,_) = &node.node_type {
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
                    let base = memo.get(&(main_asg.id, *base_id)).unwrap();
                    let exp_node_id = *exp_id;

                    // --- НАЧАЛО ИСПРАВЛЕНИЯ ---
                    // Проверяем, является ли степень константой 2.0.
                    let is_square_op = main_asg.get_node(exp_node_id)
                        .map(|node| match &node.node_type {
                            // ИСПРАВЛЕНО: Правильный способ проверить скаляр и его значение.
                            NodeType::Literal(Value::Tensor(t)) => t.ndim() == 0 && t.iter().next().map_or(false, |&v| v == 2.0),
                            _ => false,
                        })
                        .unwrap_or(false);

                    let shader = if is_square_op {
                        // Если это возведение в квадрат, используем численно стабильное умножение.
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
                        // В противном случае используем общую функцию pow.
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
                        let exp = memo.get(&(main_asg.id, exp_node_id)).unwrap();
                        self.dispatch_shader(&shader, shape, &[base, exp])?
                    }
                    // --- КОНЕЦ ИСПРАВЛЕНИЯ ---
                }

                NodeType::MatrixMultiply(a, b) => {
                    let lhs = memo.get(&(main_asg.id, *a)).unwrap();
                    let rhs = memo.get(&(main_asg.id, *b)).unwrap();

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
                        m = m, k = k, n = n, batch_count = batch_count
                    );
                    
                    let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("batched_matmul_shader"),
                        source: wgpu::ShaderSource::Wgsl(shader.into()),
                    });

                    let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("batched_matmul_pipeline"),
                        layout: None,
                        module: &module,
                        entry_point: "main",
                    });

                    let output_len = shape.iter().product::<usize>();
                    let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("output_buffer"),
                        size: (output_len * std::mem::size_of::<f32>()) as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    let bind_group_layout = pipeline.get_bind_group_layout(0);
                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("bind_group"),
                        layout: &bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: lhs.buffer.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: rhs.buffer.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
                        ],
                    });

                    let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    compute_pass.set_pipeline(&pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    
                    let workgroup_x = ((n as u32) + 7) / 8;
                    let workgroup_y = ((m as u32) + 7) / 8;
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
                    let input = memo.get(&(main_asg.id, *x)).unwrap();
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
                    let input = memo.get(&(main_asg.id, *x)).unwrap();
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
                    let input = memo.get(&(main_asg.id, *x)).unwrap();
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
                    let input = memo.get(&(main_asg.id, *x)).unwrap();
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
                    let input = memo.get(&(main_asg.id, *x)).unwrap();
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
                    let input = memo.get(&(main_asg.id, *x)).unwrap();
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
                    let input = memo.get(&(main_asg.id, *x)).unwrap();
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
                    let input = memo.get(&(main_asg.id, *x)).unwrap();
                    let shader = format!(r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let idx = id.x;
                            if (idx >= arrayLength(&o)) {{ return; }}
                            o[idx] = clamp(x[idx], {min_val}f, {max_val}f);
                        }}
                    "#, min_val = min_val, max_val = max_val);
                    self.dispatch_shader(&shader, shape, &[input])?
                }

                NodeType::LeakyReLU(x, alpha) => {
                    let input = memo.get(&(main_asg.id, *x)).unwrap();
                    let shader = format!(r#"
                        @group(0) @binding(0) var<storage, read> x: array<f32>;
                        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                            let idx = id.x;
                            if (idx >= arrayLength(&o)) {{ return; }}
                            let val = x[idx];
                            o[idx] = select({alpha}f * val, val, val > 0.0);
                        }}
                    "#, alpha = alpha);
                    self.dispatch_shader(&shader, shape, &[input])?
                }

                NodeType::GELU(x) => {
                    let input = memo.get(&(main_asg.id, *x)).unwrap();
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
                    let input = memo.get(&(main_asg.id, *x)).unwrap();
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
                    let input = memo.get(&(main_asg.id, *x)).unwrap();
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
                    let input = memo.get(&(main_asg.id, *x)).unwrap();
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
                    let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("softmax_shader"),
                        source: wgpu::ShaderSource::Wgsl(shader.into()),
                    });

                    let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("softmax_pipeline"),
                        layout: None,
                        module: &module,
                        entry_point: "main",
                    });

                    let output_len = shape.iter().product::<usize>();
                    let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("softmax_output"),
                        size: (output_len * std::mem::size_of::<f32>()) as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    let bind_group_layout = pipeline.get_bind_group_layout(0);
                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("softmax_bind_group"),
                        layout: &bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: input.buffer.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
                        ],
                    });

                    let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
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
                    let input = memo.get(&(main_asg.id, *x)).unwrap();
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
                    let input = memo.get(&(main_asg.id, *x)).unwrap();
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
                    let input = memo.get(&(main_asg.id, *x)).unwrap();
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
                    let data_tensor = memo.get(&(main_asg.id, *data_id)).unwrap();
                    GpuTensor {
                        buffer: self.copy_buffer(&data_tensor.buffer),
                        shape: shape.clone(),
                    }
                }

                NodeType::Broadcast(source_id, _target_id) => {
                    let source = memo.get(&(main_asg.id, *source_id)).unwrap();
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
                    let input = memo.get(&(main_asg.id, *data_id)).unwrap();
                    let rank = input.shape.len();
                    let total_out = shape.iter().product::<usize>();

                    let mut strides_out = vec![0; rank];
                    strides_out[rank - 1] = 1;
                    for i in (0..rank - 1).rev() {
                        strides_out[i] = strides_out[i + 1] * shape[i + 1];
                    }

                    let mut strides_in = vec![0; rank];
                    strides_in[rank-1] = 1;
                    for i in (0..rank - 1).rev() {
                        strides_in[i] = strides_in[i + 1] * input.shape[i + 1];
                    }
                    
                    let coord_code = (0..rank).map(|i| {
                        format!("let c{i} = rem / {s}u; coords[{i}] = c{i}; rem = rem - c{i} * {s}u;", s = strides_out[i])
                    }).collect::<Vec<_>>().join("\n                            ");

                    let index_code = (0..rank).map(|i| {
                        format!("idx = idx + in_coords[{i}] * {s}u;", s = strides_in[i])
                    }).collect::<Vec<_>>().join("\n                            ");
                    
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
                        rank = rank, ax1 = ax1, ax2 = ax2, total_out = total_out,
                        coord_code = coord_code, index_code = index_code
                    );
                    self.dispatch_shader(&shader, shape, &[input])?
                }

                NodeType::External { source_asg_id, source_node_id, .. } => {
                    let tensor = memo.get(&(*source_asg_id, *source_node_id)).ok_or(
                        RuntimeError::NodeNotFound(*source_node_id, *source_asg_id)
                    )?;
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
                    let grad = memo.get(&(main_asg.id, *grad_id)).unwrap();
                    let target = memo.get(&(main_asg.id, *target_id)).unwrap();
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
                        let shader = format!(r#"
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
                        "#, out_size = out_size, repeat = repeat_factor);
                        self.dispatch_shader(&shader, target_shape, &[grad])?
                    }
                }

                // Conv2d: Direct convolution implementation
                NodeType::Conv2d { input, weight, bias, stride, padding, dilation, groups } => {
                    if *groups != 1 {
                        return Err(RuntimeError::UnimplementedOperation(
                            "Conv2d with groups != 1 not yet supported on GPU".to_string()
                        ));
                    }
                    if *dilation != (1, 1) {
                        return Err(RuntimeError::UnimplementedOperation(
                            "Conv2d with dilation != (1,1) not yet supported on GPU".to_string()
                        ));
                    }

                    let input_tensor = memo.get(&(main_asg.id, *input)).unwrap();
                    let weight_tensor = memo.get(&(main_asg.id, *weight)).unwrap();

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

                    let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("conv2d_shader"),
                        source: wgpu::ShaderSource::Wgsl(shader.into()),
                    });

                    let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("conv2d_pipeline"),
                        layout: None,
                        module: &module,
                        entry_point: "main",
                    });

                    let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("conv2d_output"),
                        size: (output_size * std::mem::size_of::<f32>()) as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    let bind_group_layout = pipeline.get_bind_group_layout(0);
                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("conv2d_bind_group"),
                        layout: &bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: input_tensor.buffer.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: weight_tensor.buffer.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
                        ],
                    });

                    let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    compute_pass.set_pipeline(&pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    compute_pass.dispatch_workgroups((output_size as u32 + 63) / 64, 1, 1);
                    drop(compute_pass);
                    self.queue.submit(Some(encoder.finish()));

                    // Handle bias if present
                    let result = if let Some(bias_id) = bias {
                        let bias_tensor = memo.get(&(main_asg.id, *bias_id)).unwrap();

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

                        let bias_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some("conv2d_bias_shader"),
                            source: wgpu::ShaderSource::Wgsl(bias_shader.into()),
                        });

                        let bias_pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some("conv2d_bias_pipeline"),
                            layout: None,
                            module: &bias_module,
                            entry_point: "main",
                        });

                        let final_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("conv2d_final_output"),
                            size: (output_size * std::mem::size_of::<f32>()) as u64,
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });

                        let bias_bind_group_layout = bias_pipeline.get_bind_group_layout(0);
                        let bias_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("conv2d_bias_bind_group"),
                            layout: &bias_bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry { binding: 0, resource: output_buffer.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 1, resource: bias_tensor.buffer.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 2, resource: final_buffer.as_entire_binding() },
                            ],
                        });

                        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                        compute_pass.set_pipeline(&bias_pipeline);
                        compute_pass.set_bind_group(0, &bias_bind_group, &[]);
                        compute_pass.dispatch_workgroups((output_size as u32 + 63) / 64, 1, 1);
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
            let tensor = memo.get(&(main_asg.id, node_id)).unwrap();
            let buffer = self.copy_buffer(&tensor.buffer);
            outputs.push(GpuTensor {
                buffer,
                shape: tensor.shape.clone(),
            });
        }

        Ok((outputs, memo))
    }

    fn retrieve_data(
        &self,
        device_data: &[Self::DeviceData],
    ) -> Result<Vec<Value>, RuntimeError> {
        let mut result = Vec::new();
        for tensor in device_data {
            let buffer_size = tensor.buffer.size();
            let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encoder.copy_buffer_to_buffer(&tensor.buffer, 0, &staging_buffer, 0, buffer_size);
            self.queue.submit(Some(encoder.finish()));

            let buffer_slice = staging_buffer.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
            self.device.poll(wgpu::Maintain::Wait);
            pollster::block_on(receiver.receive()).unwrap().unwrap();

            let data = buffer_slice.get_mapped_range();
            let slice: &[f32] = bytemuck::cast_slice(&data);
            let array = ndarray::ArrayD::from_shape_vec(tensor.shape.clone(), slice.to_vec()).unwrap();
            drop(data);
            staging_buffer.unmap();

            result.push(Value::Tensor(array));
        }
        Ok(result)
    }
}
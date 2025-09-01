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

                NodeType::Add(a, b) => {
                    let lhs = memo.get(&(main_asg.id, *a)).unwrap();
                    let rhs = memo.get(&(main_asg.id, *b)).unwrap();
                    let shader = r#"
                        @group(0) @binding(0) var<storage, read> a: array<f32>;
                        @group(0) @binding(1) var<storage, read> b: array<f32>;
                        @group(0) @binding(2) var<storage, read_write> o: array<f32>;
                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let idx = id.x;
                            if (idx >= arrayLength(&o)) { return; }
                            o[idx] = a[idx] + b[idx];
                        }
                    "#;
                    self.dispatch_shader(shader, shape, &[lhs, rhs])?
                }

                NodeType::Multiply(a, b) => {
                    let lhs = memo.get(&(main_asg.id, *a)).unwrap();
                    let rhs = memo.get(&(main_asg.id, *b)).unwrap();
                    let shader = r#"
                        @group(0) @binding(0) var<storage, read> a: array<f32>;
                        @group(0) @binding(1) var<storage, read> b: array<f32>;
                        @group(0) @binding(2) var<storage, read_write> o: array<f32>;
                        @compute @workgroup_size(64)
                        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                            let idx = id.x;
                            if (idx >= arrayLength(&o)) { return; }
                            o[idx] = a[idx] * b[idx];
                        }
                    "#;
                    self.dispatch_shader(shader, shape, &[lhs, rhs])?
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

                NodeType::Sum(x) => {
                    let input = memo.get(&(main_asg.id, *x)).unwrap();
                    let len = shape.iter().product::<usize>();
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
    let len = shape.iter().product::<usize>();
    let last_dim = input.shape.last().copied().unwrap_or(1);
    let outer = len / last_dim;

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
        outer = outer,
        last_dim = last_dim
    );

    self.dispatch_shader(&shader, shape, &[input])?
}

NodeType::Subtract(a, b) => {
    let lhs = memo.get(&(main_asg.id, *a)).unwrap();
    let rhs = memo.get(&(main_asg.id, *b)).unwrap();
    let shader = r#"
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let idx = id.x;
            if (idx >= arrayLength(&o)) { return; }
            o[idx] = a[idx] - b[idx];
        }
    "#;
    self.dispatch_shader(shader, shape, &[lhs, rhs])?
}

NodeType::Variance(x) => {
    let input = memo.get(&(main_asg.id, *x)).unwrap();
    let len = shape.iter().product::<usize>();
    let last_dim = input.shape.last().copied().unwrap_or(1);
    let outer = len / last_dim;

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
            var sum_sq = 0.0;
            for (var i = 0u; i < {last_dim}u; i = i + 1u) {{
                let val = x[offset + i];
                sum = sum + val;
                sum_sq = sum_sq + val * val;
            }}
            let mean = sum / f32({last_dim}u);
            let mean_sq = sum_sq / f32({last_dim}u);
            o[outer_idx] = mean_sq - mean * mean;
        }}
        "#,
        outer = outer,
        last_dim = last_dim
    );

    self.dispatch_shader(&shader, shape, &[input])?
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

NodeType::Divide(a, b) => {
    let lhs = memo.get(&(main_asg.id, *a)).unwrap();
    let rhs = memo.get(&(main_asg.id, *b)).unwrap();
    let shader = r#"
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let idx = id.x;
            if (idx >= arrayLength(&o)) { return; }
            o[idx] = a[idx] / b[idx];
        }
    "#;
    self.dispatch_shader(shader, shape, &[lhs, rhs])?
}

NodeType::Softmax(x) => {
    let input = memo.get(&(main_asg.id, *x)).unwrap();
    let last_dim = input.shape.last().copied().unwrap_or(1);
    let outer = shape.iter().product::<usize>() / last_dim;

    let shader = format!(
        r#"
        @group(0) @binding(0) var<storage, read> x: array<f32>;
        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
            let outer_idx = id.x;
            if (outer_idx >= {outer}u) {{ return; }}
            let offset = outer_idx * {last_dim}u;

            var max_val = x[offset];
            for (var i = 1u; i < {last_dim}u; i = i + 1u) {{
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
        outer = outer,
        last_dim = last_dim
    );
    self.dispatch_shader(&shader, shape, &[input])?
}

NodeType::MatrixMultiply(a, b) => {
    let lhs = memo.get(&(main_asg.id, *a)).unwrap();
    let rhs = memo.get(&(main_asg.id, *b)).unwrap();

    let m = lhs.shape[lhs.shape.len() - 2];
    let k = lhs.shape[lhs.shape.len() - 1];
    let n = rhs.shape[rhs.shape.len() - 1];

    let shader = format!(
        r#"
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
            let m = {m}u;
            let k = {k}u;
            let n = {n}u;
            let row = id.y;
            let col = id.x;
            if (row >= m || col >= n) {{ return; }}

            var sum = 0.0;
            for (var i = 0u; i < k; i = i + 1u) {{
                sum = sum + a[row * k + i] * b[i * n + col];
            }}
            o[row * n + col] = sum;
        }}
        "#,
        m = m,
        k = k,
        n = n
    );

    self.dispatch_shader(&shader, shape, &[lhs, rhs])?
}

NodeType::Reshape(data_id, shape_id) => {
    let data_tensor = memo.get(&(main_asg.id, *data_id)).unwrap();

    // Получаем shape из уже вычисленного shape-узла
    let shape_node = main_asg.get_node(*shape_id).unwrap();
    let new_shape = match &shape_node.node_type {
        NodeType::Literal(Value::Tensor(t)) => t.iter().map(|&x| x as usize).collect::<Vec<_>>(),
        _ => {
            return Err(RuntimeError::UnimplementedOperation(
                "Reshape shape must be literal".to_string(),
            ))
        }
    };

    let out_len = new_shape.iter().product::<usize>();
    if data_tensor.shape.iter().product::<usize>() != out_len {
        return Err(RuntimeError::ShapeError(format!(
            "Reshape length mismatch: {} != {}",
            data_tensor.shape.iter().product::<usize>(),
            out_len
        )));
    }

    let shader = r#"
        @group(0) @binding(0) var<storage, read> src: array<f32>;
        @group(0) @binding(1) var<storage, read_write> dst: array<f32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let idx = id.x;
            if (idx >= arrayLength(&dst)) { return; }
            dst[idx] = src[idx];
        }
    "#;

    self.dispatch_shader(shader, &new_shape, &[data_tensor])?
}

NodeType::Broadcast(source_id, target_id) => {
    let source = memo.get(&(main_asg.id, *source_id)).unwrap();
    let target = memo.get(&(main_asg.id, *target_id)).unwrap();
    let out_len = shape.iter().product::<usize>();

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
    let total = shape.iter().product::<usize>();

    let shader = format!(
        r#"
        @group(0) @binding(0) var<storage, read> src: array<f32>;
        @group(0) @binding(1) var<storage, read_write> dst: array<f32>;

        const RANK: u32 = {rank}u;

        fn index_to_coords(index: u32) -> array<u32, {rank}> {{
            var coords: array<u32, {rank}>;
            var rem = index;
            {coord_code}
            return coords;
        }}

        fn coords_to_index(coords: array<u32, {rank}>) -> u32 {{
            var idx = 0u;
            {index_code}
            return idx;
        }}

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
            let out_idx = id.x;
            if (out_idx >= {total}u) {{ return; }}

            var coords = index_to_coords(out_idx);
            let tmp = coords[{ax1}];
            coords[{ax1}] = coords[{ax2}];
            coords[{ax2}] = tmp;

            let in_idx = coords_to_index(coords);
            dst[out_idx] = src[in_idx];
        }}
        "#,
        rank = rank,
        ax1 = ax1,
        ax2 = ax2,
        total = total,
        coord_code = (0..rank)
            .rev()
            .map(|i| {
                let stride: usize = shape.iter().skip(i + 1).product();
                format!(
                    "coords[{i}] = rem / {stride}u;\n            rem = rem % {stride}u;"
                )
            })
            .collect::<Vec<_>>()
            .join("\n            "),
        index_code = (0..rank)
            .map(|i| {
                let stride: usize = shape.iter().skip(i + 1).product();
                format!("idx = idx * {stride}u + coords[{i}];")
            })
            .collect::<Vec<_>>()
            .join("\n            ")
    );

    self.dispatch_shader(&shader, shape, &[input])?
}

NodeType::Power(base_id, exp_id) => {
    let base = memo.get(&(main_asg.id, *base_id)).unwrap();
    let exp  = memo.get(&(main_asg.id, *exp_id)).unwrap();
    let exp_len = exp.shape.iter().product::<usize>();

    if exp_len != 1 {
        return Err(RuntimeError::ShapeError(
            "Power supports only scalar exponent".to_string(),
        ));
    }

    let shader = r#"
        @group(0) @binding(0) var<storage, read> base: array<f32>;
        @group(0) @binding(1) var<storage, read> exp: array<f32>;
        @group(0) @binding(2) var<storage, read_write> out: array<f32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let idx = id.x;
            if (idx >= arrayLength(&out)) { return; }
            out[idx] = pow(base[idx], exp[0]);
        }
    "#;

    self.dispatch_shader(shader, shape, &[base, exp])?
}

NodeType::External {
    source_asg_id,
    source_node_id,
    ..
} => {
    // External узел должен быть уже в memo
    let tensor = memo
        .get(&(*source_asg_id, *source_node_id))
        .ok_or(RuntimeError::NodeNotFound(
            *source_node_id,
            *source_asg_id,
        ))?;
    // Просто копируем ссылку (или клонируем буфер, если нужно)
    GpuTensor {
        buffer: self.copy_buffer(&tensor.buffer),
        shape: tensor.shape.clone(),
    }
}

NodeType::GreaterThan(a, b) => {
    let lhs = memo.get(&(main_asg.id, *a)).unwrap();
    let rhs = memo.get(&(main_asg.id, *b)).unwrap();

    let shader = r#"
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let idx = id.x;
            if (idx >= arrayLength(&o)) { return; }
            o[idx] = select(0.0, 1.0, a[idx] > b[idx]);
        }
    "#;

    self.dispatch_shader(shader, shape, &[lhs, rhs])?
}

                _ => {
                    return Err(RuntimeError::UnimplementedOperation(format!(
                        "node type not supported: {:?}",
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
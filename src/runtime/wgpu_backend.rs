//! Модуль, реализующий бэкенд для выполнения ASG на GPU с использованием wgpu.

use super::backend::{Backend, RuntimeError};
use crate::asg::{Asg, AsgId, NodeId, NodeType, Shape, Value};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

// --- Вспомогательные структуры ---

/// Представление тензора, находящегося в памяти GPU.
#[derive(Debug)]
pub struct GpuTensor {
    /// Буфер на GPU, хранящий данные тензора в виде плоского массива f32.
    buffer: wgpu::Buffer,
    /// Форма тензора, необходимая для правильной интерпретации данных.
    shape: Shape,
}

/// Исполнительный бэкенд, работающий на GPU через wgpu.
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

// --- Реализация бэкенда ---

impl WgpuBackend {
    /// Создает новый экземпляр WgpuBackend.
    /// Эта функция асинхронная, так как инициализация GPU требует времени.
    pub async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Не удалось найти подходящий GPU адаптер.");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("WGPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None, // trace_path
            )
            .await
            .expect("Не удалось получить логическое устройство GPU.");

        Self { device, queue }
    }
}

impl Backend for WgpuBackend {
    type DeviceData = GpuTensor;

    /// Копирует данные с CPU (Value) в память GPU (GpuTensor).
    fn load_data(
        &self,
        data: &HashMap<String, Value>,
    ) -> Result<HashMap<String, Self::DeviceData>, RuntimeError> {
        let mut device_data = HashMap::new();
        for (name, value) in data {
            if let Value::Tensor(tensor) = value {
                let bytes: &[u8] = bytemuck::cast_slice(tensor.as_slice().unwrap());
                let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(name),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });
                let gpu_tensor = GpuTensor {
                    buffer,
                    shape: tensor.shape().to_vec(),
                };
                device_data.insert(name.clone(), gpu_tensor);
            }
        }
        Ok(device_data)
    }

    /// Выполняет граф на GPU.
    fn run(
        &self,
        main_asg: &Asg,
        inputs: &HashMap<String, Self::DeviceData>,
        linked_graphs: &[&Asg],
    ) -> Result<Vec<Self::DeviceData>, RuntimeError> {
        // Создаем хранилище для всех графов, чтобы External узлы могли их найти
        let mut all_graphs = HashMap::new();
        all_graphs.insert(main_asg.id, main_asg);
        for g in linked_graphs {
            all_graphs.insert(g.id, *g);
        }

        let mut memo: HashMap<(AsgId, NodeId), GpuTensor> = HashMap::new();
        
        // Клонируем входы в memo, чтобы External узлы могли их найти
        for (name, gpu_tensor) in inputs.iter() {
            // Ищем соответствующий узел Input/Parameter в главном графе
            if let Some(node_id) = main_asg.nodes.values()
                .find(|n| match &n.node_type {
                    NodeType::Input{name: n_name} | NodeType::Parameter{name: n_name} => n_name == name,
                    _ => false
                }).map(|n| n.id) {
                
                // Копируем буфер, чтобы избежать проблем с владением
                let copied_buffer = self.copy_buffer(&gpu_tensor.buffer);
                memo.insert((main_asg.id, node_id), GpuTensor { buffer: copied_buffer, shape: gpu_tensor.shape.clone() });
            }
        }

        let sorted_nodes = crate::analysis::shape_inference::ShapeInference::topological_sort(main_asg)
            .map_err(|e| RuntimeError::ShapeError(format!("Topological sort failed: {:?}", e)))?;

        for node_id in sorted_nodes {
            let node = main_asg.get_node(node_id).unwrap();
            let output_shape = node.shape.as_ref().expect("Shape info missing!").clone();

            let output_tensor = match &node.node_type {
                NodeType::Input { .. } | NodeType::Parameter { .. } => {
                    // Уже обработано выше, просто пропускаем
                    continue;
                }
                NodeType::Literal(value) => {
                    if let Value::Tensor(t) = value {
                        let bytes: &[u8] = bytemuck::cast_slice(t.as_slice().unwrap());
                        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("literal_{}", node.id)),
                            contents: bytes,
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        });
                        GpuTensor { buffer, shape: t.shape().to_vec() }
                    } else {
                        return Err(RuntimeError::TypeError{ expected: "Tensor Literal".to_string(), actual: "Other".to_string() });
                    }
                }
                NodeType::External { source_asg_id, source_node_id, .. } => {
                    let source_tensor = memo.get(&(*source_asg_id, *source_node_id)).unwrap();
                    let copied_buffer = self.copy_buffer(&source_tensor.buffer);
                    GpuTensor { buffer: copied_buffer, shape: source_tensor.shape.clone() }
                }

                NodeType::Add(l, r) => self.execute_binary_elementwise("add", &output_shape, memo.get(&(main_asg.id, *l)).unwrap(), memo.get(&(main_asg.id, *r)).unwrap())?,
                NodeType::Subtract(l, r) => self.execute_binary_elementwise("sub", &output_shape, memo.get(&(main_asg.id, *l)).unwrap(), memo.get(&(main_asg.id, *r)).unwrap())?,
                NodeType::Multiply(l, r) => self.execute_binary_elementwise("mul", &output_shape, memo.get(&(main_asg.id, *l)).unwrap(), memo.get(&(main_asg.id, *r)).unwrap())?,
                NodeType::Divide(l, r) => self.execute_binary_elementwise("div", &output_shape, memo.get(&(main_asg.id, *l)).unwrap(), memo.get(&(main_asg.id, *r)).unwrap())?,
                NodeType::GreaterThan(l, r) => self.execute_binary_elementwise("gt", &output_shape, memo.get(&(main_asg.id, *l)).unwrap(), memo.get(&(main_asg.id, *r)).unwrap())?,

                NodeType::ReLU(id) => self.execute_unary_elementwise("relu", &output_shape, memo.get(&(main_asg.id, *id)).unwrap())?,
                NodeType::Sqrt(id) => self.execute_unary_elementwise("sqrt", &output_shape, memo.get(&(main_asg.id, *id)).unwrap())?,
                
                NodeType::Mean(id) => self.execute_reduction("mean", &output_shape, memo.get(&(main_asg.id, *id)).unwrap())?,
                NodeType::Sum(id) => self.execute_reduction("sum", &output_shape, memo.get(&(main_asg.id, *id)).unwrap())?,
                NodeType::Softmax(id) => self.execute_softmax(&output_shape, memo.get(&(main_asg.id, *id)).unwrap())?,

                NodeType::MatrixMultiply(l, r) => self.execute_matmul(&output_shape, memo.get(&(main_asg.id, *l)).unwrap(), memo.get(&(main_asg.id, *r)).unwrap())?,
                
                NodeType::Transpose(id, axis1, axis2) => self.execute_transpose(&output_shape, memo.get(&(main_asg.id, *id)).unwrap(), *axis1, *axis2)?,

                NodeType::Reshape(id, _shape_id) => {
                    // Reshape - это просто смена метаданных, буфер остается тем же
                    let input_tensor = memo.get(&(main_asg.id, *id)).unwrap();
                    let new_buffer = self.copy_buffer(&input_tensor.buffer);
                    GpuTensor { buffer: new_buffer, shape: output_shape }
                }

                NodeType::Broadcast(source_id, target_id) => {
                    let source = memo.get(&(main_asg.id, *source_id)).unwrap();
                    let target = memo.get(&(main_asg.id, *target_id)).unwrap();
                    self.execute_broadcast(&output_shape, source, target)?
                }
                
                _ => return Err(RuntimeError::UnimplementedOperation(format!("{:?}", node.node_type))),
            };

            memo.insert((main_asg.id, node_id), output_tensor);
        }
        
        let results = main_asg.outputs.iter()
            .map(|id| memo.remove(&(main_asg.id, *id)).unwrap())
            .collect();

        Ok(results)
    }

    /// Копирует данные с GPU (GpuTensor) обратно на CPU (Value).
    fn retrieve_data(&self, device_data: &[Self::DeviceData]) -> Result<Vec<Value>, RuntimeError> {
        let mut cpu_values = Vec::new();

        for tensor in device_data {
            let buffer_size = tensor.buffer.size();
            
            let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Buffer"),
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
            let result: &[f32] = bytemuck::cast_slice(&data);
            
            let array = ndarray::ArrayD::from_shape_vec(tensor.shape.clone(), result.to_vec()).unwrap();
            
            drop(data);
            staging_buffer.unmap();
            
            cpu_values.push(Value::Tensor(array));
        }

        Ok(cpu_values)
    }
}

// --- Реализации Шейдеров ---

impl WgpuBackend {
    fn copy_buffer(&self, source_buffer: &wgpu::Buffer) -> wgpu::Buffer {
        let size = source_buffer.size();
        let new_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("copied_buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(source_buffer, 0, &new_buffer, 0, size);
        self.queue.submit(Some(encoder.finish()));
        new_buffer
    }

    fn execute_unary_elementwise(&self, op: &str, output_shape: &Shape, input: &GpuTensor) -> Result<GpuTensor, RuntimeError> {
        let op_code = match op {
            "relu" => "return max(val, 0.0);",
            "sqrt" => "return sqrt(val);",
            _ => panic!("Неизвестный унарный шейдер"),
        };
        let shader_code = format!(r#"
            fn op(val: f32) -> f32 {{
                {op_code}
            }}

            @group(0) @binding(0) var<storage, read> input_buf: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output_buf: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let index = global_id.x;
                if (index >= arrayLength(&input_buf)) {{ return; }}
                output_buf[index] = op(input_buf[index]);
            }}
        "#, op_code = op_code);
        self.dispatch_shader(&shader_code, output_shape, &[input])
    }
    
    fn execute_binary_elementwise(&self, op: &str, output_shape: &Shape, lhs: &GpuTensor, rhs: &GpuTensor) -> Result<GpuTensor, RuntimeError> {
        let op_code = match op {
            "add" => "return lhs + rhs;",
            "sub" => "return lhs - rhs;",
            "mul" => "return lhs * rhs;",
            "div" => "return lhs / rhs;",
            "gt"  => "if (lhs > rhs) {{ return 1.0; }} else {{ return 0.0; }}",
            _ => panic!("Неизвестный бинарный шейдер"),
        };
        
        let rhs_is_scalar = rhs.shape.iter().product::<usize>() == 1;
        let rhs_access = if rhs_is_scalar { "rhs_buf[0]" } else { "rhs_buf[index]" };

        let shader_code = format!(r#"
            fn op(lhs: f32, rhs: f32) -> f32 {{
                {op_code}
            }}

            @group(0) @binding(0) var<storage, read> lhs_buf: array<f32>;
            @group(0) @binding(1) var<storage, read> rhs_buf: array<f32>;
            @group(0) @binding(2) var<storage, read_write> output_buf: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let index = global_id.x;
                if (index >= arrayLength(&lhs_buf)) {{ return; }}
                output_buf[index] = op(lhs_buf[index], {rhs_access});
            }}
        "#, op_code = op_code, rhs_access = rhs_access);
        self.dispatch_shader(&shader_code, output_shape, &[lhs, rhs])
    }

    fn execute_reduction(&self, op: &str, output_shape: &Shape, input: &GpuTensor) -> Result<GpuTensor, RuntimeError> {
        let last_dim = *input.shape.last().unwrap_or(&1);
        let outer_dims: usize = input.shape.iter().rev().skip(1).product();

        let op_init = match op { "sum" | "mean" => "0.0", _ => panic!("Unknown reduction op") };
        let op_accum = "sum = sum + val;";
        let op_final = match op { "sum" => "sum", "mean" => "sum / last_dim_f32", _ => "" };

        let shader_code = format!(r#"
            @group(0) @binding(0) var<storage, read> input_buf: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output_buf: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let outer_index = global_id.x;
                let last_dim: u32 = {last_dim}u;
                let last_dim_f32: f32 = {last_dim_f};
                
                if (outer_index >= {outer_dims}u) {{ return; }}

                var sum: f32 = {op_init};
                for (var i: u32 = 0u; i < last_dim; i = i + 1u) {{
                    let val = input_buf[outer_index * last_dim + i];
                    {op_accum}
                }}
                
                output_buf[outer_index] = {op_final};
            }}
        "#, 
        last_dim = last_dim,
        last_dim_f = last_dim as f32,
        outer_dims = outer_dims, 
        op_init = op_init, 
        op_accum = op_accum, 
        op_final = op_final
        );

        self.dispatch_shader(&shader_code, output_shape, &[input])
    }

    fn execute_matmul(&self, output_shape: &Shape, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, RuntimeError> {
        if a.shape.len() == 2 && b.shape.len() == 2 { // 2D Matmul
            let m = a.shape[0]; let k = a.shape[1]; let n = b.shape[1];
            let shader = format!(r#"
                @group(0) @binding(0) var<storage, read> a: array<f32>;
                @group(0) @binding(1) var<storage, read> b: array<f32>;
                @group(0) @binding(2) var<storage, read_write> out: array<f32>;
                
                @compute @workgroup_size(8, 8)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                    let M: u32 = {m}u; let K: u32 = {k}u; let N: u32 = {n}u;
                    let r = global_id.y; let c = global_id.x;
                    if (r >= M || c >= N) {{ return; }}
                    var sum = 0.0;
                    for (var i: u32 = 0u; i < K; i = i + 1u) {{
                        sum = sum + a[r * K + i] * b[i * N + c];
                    }}
                    out[r * N + c] = sum;
                }}
            "#, m=m, k=k, n=n);
            self.dispatch_shader(&shader, output_shape, &[a, b])
        } else if a.shape.len() == 4 && b.shape.len() == 4 { // 4D Batched Matmul
            let (b0, b1, m, k) = (a.shape[0], a.shape[1], a.shape[2], a.shape[3]);
            let n = b.shape[3];
            let shader = format!(r#"
                @group(0) @binding(0) var<storage, read> a: array<f32>;
                @group(0) @binding(1) var<storage, read> b: array<f32>;
                @group(0) @binding(2) var<storage, read_write> out: array<f32>;
                
                @compute @workgroup_size(8, 8)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                    let B0: u32 = {b0}u; let B1: u32 = {b1}u; let M: u32 = {m}u; let K: u32 = {k}u; let N: u32 = {n}u;
                    let b0_idx = global_id.z / B1;
                    let b1_idx = global_id.z % B1;
                    let r = global_id.y; let c = global_id.x;
                    if (r >= M || c >= N || b0_idx >= B0 || b1_idx >= B1) {{ return; }}

                    let a_offset = (b0_idx * B1 + b1_idx) * M * K;
                    let b_offset = (b0_idx * B1 + b1_idx) * K * N;
                    let out_offset = (b0_idx * B1 + b1_idx) * M * N;

                    var sum = 0.0;
                    for (var i: u32 = 0u; i < K; i = i + 1u) {{
                        sum = sum + a[a_offset + r * K + i] * b[b_offset + i * N + c];
                    }}
                    out[out_offset + r * N + c] = sum;
                }}
            "#, b0=b0, b1=b1, m=m, k=k, n=n);
            self.dispatch_shader_3d(&shader, output_shape, &[a, b])
        } else {
             Err(RuntimeError::UnimplementedOperation(format!("Matmul for dims {:?} and {:?}", a.shape, b.shape)))
        }
    }
    
    fn execute_softmax(&self, output_shape: &Shape, input: &GpuTensor) -> Result<GpuTensor, RuntimeError> {
        let last_dim = *input.shape.last().unwrap_or(&1);
        let outer_dims: usize = input.shape.iter().rev().skip(1).product();

        let shader_code = format!(r#"
            @group(0) @binding(0) var<storage, read> input_buf: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output_buf: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let outer_index = global_id.x;
                let last_dim: u32 = {last_dim}u;
                if (outer_index >= {outer_dims}u) {{ return; }}

                let offset = outer_index * last_dim;
                
                var max_val = -3.402823466e+38; // f32.min
                for (var i: u32 = 0u; i < last_dim; i = i + 1u) {{
                    max_val = max(max_val, input_buf[offset + i]);
                }}

                var sum: f32 = 0.0;
                for (var i: u32 = 0u; i < last_dim; i = i + 1u) {{
                    let val = exp(input_buf[offset + i] - max_val);
                    output_buf[offset + i] = val;
                    sum = sum + val;
                }}
                
                for (var i: u32 = 0u; i < last_dim; i = i + 1u) {{
                    output_buf[offset + i] = output_buf[offset + i] / sum;
                }}
            }}
        "#, last_dim=last_dim, outer_dims=outer_dims);
        self.dispatch_shader(&shader_code, output_shape, &[input])
    }

    fn execute_transpose(&self, output_shape: &Shape, input: &GpuTensor, axis1: usize, axis2: usize) -> Result<GpuTensor, RuntimeError> {
        let rank = input.shape.len();
        if rank < 2 { return Err(RuntimeError::ShapeError("Transpose requires rank >= 2".to_string())); }
        
        let shader = format!(r#"
            @group(0) @binding(0) var<storage, read> input_buf: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output_buf: array<f32>;

            {dims_vars}

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let out_idx = global_id.x;
                if (out_idx >= arrayLength(&output_buf)) {{ return; }}

                var out_coords: array<u32, {rank}>;
                var temp_idx = out_idx;
                {out_coords_calc}

                var in_coords = out_coords;
                in_coords[{axis1}u] = out_coords[{axis2}u];
                in_coords[{axis2}u] = out_coords[{axis1}u];
                
                var in_idx: u32 = 0u;
                {in_idx_calc}
                
                output_buf[out_idx] = input_buf[in_idx];
            }}
        "#, 
        rank = rank, 
        axis1 = axis1, 
        axis2 = axis2,
        dims_vars = (0..rank).map(|i| format!("let d{i}: u32 = {dim}u;", i=i, dim=input.shape[i])).collect::<Vec<_>>().join("\n"),
        out_coords_calc = (0..rank).rev().map(|i| {
            let stride: usize = output_shape.iter().skip(i + 1).product();
            format!("out_coords[{i}] = temp_idx / {stride}u; temp_idx = temp_idx % {stride}u;", i=i, stride=stride)
        }).collect::<Vec<_>>().join("\n"),
        in_idx_calc = (0..rank).map(|i| {
            let stride: usize = input.shape.iter().skip(i + 1).product();
            format!("in_idx = in_idx + in_coords[{i}] * {stride}u;", i=i, stride=stride)
        }).collect::<Vec<_>>().join("\n")
        );
        self.dispatch_shader(&shader, output_shape, &[input])
    }
    
    fn execute_broadcast(&self, output_shape: &Shape, source: &GpuTensor, target: &GpuTensor) -> Result<GpuTensor, RuntimeError> {
        let shader = r#"
            @group(0) @binding(0) var<storage, read> source_buf: array<f32>;
            @group(0) @binding(1) var<storage, read> target_buf: array<f32>;
            @group(0) @binding(2) var<storage, read_write> output_buf: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x;
                if (index >= arrayLength(&output_buf)) { return; }
                output_buf[index] = source_buf[0];
            }
        "#;
        self.dispatch_shader(shader, output_shape, &[source, target])
    }

    fn dispatch_shader(&self, shader_code: &str, output_shape: &Shape, inputs: &[&GpuTensor]) -> Result<GpuTensor, RuntimeError> {
        let output_size = output_shape.iter().product::<usize>() as u64 * 4;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shader_output_buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader_module"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });
        
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shader_pipeline"),
            layout: None,
            module: &shader_module,
            entry_point: "main",
        });

        let mut entries: Vec<wgpu::BindGroupEntry> = inputs.iter().enumerate().map(|(i, tensor)|
            wgpu::BindGroupEntry { binding: i as u32, resource: tensor.buffer.as_entire_binding() }
        ).collect();
        entries.push(wgpu::BindGroupEntry { binding: inputs.len() as u32, resource: output_buffer.as_entire_binding() });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shader_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let total_elements = output_shape.iter().product::<usize>() as u32;
            let workgroup_count = (total_elements + 63) / 64;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        Ok(GpuTensor { buffer: output_buffer, shape: output_shape.clone() })
    }

    fn dispatch_shader_3d(&self, shader_code: &str, output_shape: &Shape, inputs: &[&GpuTensor]) -> Result<GpuTensor, RuntimeError> {
        let output_size = output_shape.iter().product::<usize>() as u64 * 4;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shader_output_buffer_3d"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader_module_3d"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });
        
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shader_pipeline_3d"),
            layout: None,
            module: &shader_module,
            entry_point: "main",
        });

        let mut entries: Vec<wgpu::BindGroupEntry> = inputs.iter().enumerate().map(|(i, tensor)|
            wgpu::BindGroupEntry { binding: i as u32, resource: tensor.buffer.as_entire_binding() }
        ).collect();
        entries.push(wgpu::BindGroupEntry { binding: inputs.len() as u32, resource: output_buffer.as_entire_binding() });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shader_bind_group_3d"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let n = inputs[1].shape[3] as u32; let m = inputs[0].shape[2] as u32;
            let b0 = inputs[0].shape[0] as u32; let b1 = inputs[0].shape[1] as u32;
            let dispatch_x = (n + 7) / 8; let dispatch_y = (m + 7) / 8;
            let dispatch_z = b0 * b1;
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
        }
        self.queue.submit(Some(encoder.finish()));

        Ok(GpuTensor { buffer: output_buffer, shape: output_shape.clone() })
    }
}
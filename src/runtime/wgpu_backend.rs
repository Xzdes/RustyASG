//! Модуль, реализующий бэкенд для выполнения ASG на GPU с использованием wgpu.

use super::backend::{Backend, RuntimeError};
use crate::asg::{Asg, NodeId, NodeType, Shape, Value};
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
        // ИСПРАВЛЕНИЕ 1 (E0308): `new` ожидает ссылку.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Не удалось найти подходящий GPU адаптер.");

        // ИСПРАВЛЕНИЕ 2 (E0063, E0061): `trace` должен быть внутри `DeviceDescriptor`,
        // а `request_device` принимает только один аргумент.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("WGPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
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
        _linked_graphs: &[&Asg],
    ) -> Result<Vec<Self::DeviceData>, RuntimeError> {
        let mut memo: HashMap<NodeId, GpuTensor> = HashMap::new();
        let sorted_nodes = crate::analysis::shape_inference::ShapeInference::topological_sort(main_asg)
            .map_err(|_| RuntimeError::ShapeError("Topological sort failed".to_string()))?;

        for node_id in sorted_nodes {
            let node = main_asg.get_node(node_id).unwrap();
            let output_shape = node.shape.as_ref().expect("Shape info missing!").clone();

            let output_tensor = match &node.node_type {
                NodeType::Input { name } | NodeType::Parameter { name } => {
                    let input_tensor = inputs.get(name).unwrap();
                    let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("copy_of_{}", name)),
                        size: input_tensor.buffer.size(),
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    });
                    let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    encoder.copy_buffer_to_buffer(&input_tensor.buffer, 0, &output_buffer, 0, input_tensor.buffer.size());
                    self.queue.submit(Some(encoder.finish()));
                    GpuTensor { buffer: output_buffer, shape: input_tensor.shape.clone() }
                }

                NodeType::Add(l, r) => {
                    let lhs = memo.get(l).unwrap();
                    let rhs = memo.get(r).unwrap();
                    self.execute_binary_elementwise_shader("add", &output_shape, lhs, rhs)?
                }

                NodeType::Multiply(l, r) => {
                    let lhs = memo.get(l).unwrap();
                    let rhs = memo.get(r).unwrap();
                    self.execute_binary_elementwise_shader("mul", &output_shape, lhs, rhs)?
                }
                
                NodeType::ReLU(id) => {
                    let input = memo.get(id).unwrap();
                    self.execute_unary_elementwise_shader("relu", &output_shape, input)?
                }
                
                _ => return Err(RuntimeError::UnimplementedOperation(format!("{:?}", node.node_type))),
            };

            memo.insert(node_id, output_tensor);
        }
        
        let results = main_asg.outputs.iter()
            .map(|id| memo.remove(id).unwrap())
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

            // ИСПРАВЛЕНИЕ 3 (E0433): `MaintainBase` не найден, используем `Maintain::Wait`.
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

impl WgpuBackend {
    fn execute_unary_elementwise_shader(&self, op: &str, output_shape: &Shape, input: &GpuTensor) -> Result<GpuTensor, RuntimeError> {
        let shader_code = match op {
            "relu" => "
                @group(0) @binding(0) var<storage, read> input: array<f32>;
                @group(0) @binding(1) var<storage, read_write> output: array<f32>;

                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let index = global_id.x;
                    if (index >= arrayLength(&input)) { return; }
                    output[index] = max(input[index], 0.0);
                }
            ",
            _ => panic!("Неизвестный унарный шейдер"),
        };

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{}_output", op)),
            size: input.buffer.size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{}_shader", op)),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });
        
        // ИСПРАВЛЕНИЕ 4 (E0063, E0308): Добавлены недостающие поля `cache`, `compilation_options` и `Some()` для `entry_point`.
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{}_pipeline", op)),
            layout: None,
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{}_bind_group", op)),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroup_count = (input.buffer.size() as u32 + 63) / 64;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        Ok(GpuTensor { buffer: output_buffer, shape: output_shape.clone() })
    }
    
    fn execute_binary_elementwise_shader(&self, op: &str, output_shape: &Shape, lhs: &GpuTensor, rhs: &GpuTensor) -> Result<GpuTensor, RuntimeError> {
        let shader_code = match op {
            "add" => "output[index] = lhs[index] + rhs[index];",
            "mul" => "output[index] = lhs[index] * rhs[index];",
            _ => panic!("Неизвестный бинарный шейдер"),
        };
        let full_shader = format!("
            @group(0) @binding(0) var<storage, read> lhs: array<f32>;
            @group(0) @binding(1) var<storage, read> rhs: array<f32>;
            @group(0) @binding(2) var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let index = global_id.x;
                if (index >= arrayLength(&lhs)) {{ return; }}
                {}
            }}
        ", shader_code);

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{}_output", op)),
            size: lhs.buffer.size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{}_shader", op)),
            source: wgpu::ShaderSource::Wgsl(full_shader.into()),
        });
        
        // ИСПРАВЛЕНИЕ 5 (E0063, E0308): Добавлены недостающие поля `cache`, `compilation_options` и `Some()` для `entry_point`.
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{}_pipeline", op)),
            layout: None,
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{}_bind_group", op)),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: lhs.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: rhs.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroup_count = (lhs.buffer.size() as u32 + 63) / 64;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        Ok(GpuTensor { buffer: output_buffer, shape: output_shape.clone() })
    }
}
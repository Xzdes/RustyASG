//! Модуль, реализующий бэкенд для выполнения ASG на GPU с использованием wgpu.

use super::backend::{Backend, Memo, RuntimeError};
use crate::asg::{Asg, NodeId, NodeType, Shape, Value};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

// --- Вспомогательные структуры ---

/// Представление тензора, находящегося в памяти GPU.
#[derive(Debug)]
pub struct GpuTensor {
    buffer: wgpu::Buffer,
    shape: Shape,
}

/// Исполнительный бэкенд, работающий на GPU через wgpu.
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

// --- Реализация бэкенда ---

impl WgpuBackend {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.unwrap();
        Self { device, queue }
    }
}

impl Backend for WgpuBackend {
    type DeviceData = GpuTensor;

    fn load_data(&self, data: &HashMap<String, Value>) -> Result<HashMap<String, Self::DeviceData>, RuntimeError> {
        let mut device_data = HashMap::new();
        for (name, value) in data {
            if let Value::Tensor(tensor) = value {
                let bytes: &[u8] = bytemuck::cast_slice(tensor.as_slice().unwrap());
                let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(name),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                });
                device_data.insert(name.clone(), GpuTensor { buffer, shape: tensor.shape().to_vec() });
            }
        }
        Ok(device_data)
    }

    fn run(&self, main_asg: &Asg, mut memo: Memo<Self::DeviceData>) -> Result<(Vec<Self::DeviceData>, Memo<Self::DeviceData>), RuntimeError> {
        let sorted_nodes = crate::analysis::shape_inference::ShapeInference::topological_sort(main_asg)
            .map_err(|e| RuntimeError::ShapeError(format!("Topological sort failed: {:?}", e)))?;

        for node_id in sorted_nodes {
            if memo.contains_key(&(main_asg.id, node_id)) { continue; }

            let node = main_asg.get_node(node_id).unwrap();
            let output_shape = node.shape.as_ref().expect("Shape info missing!").clone();

            let output_tensor = match &node.node_type {
                NodeType::Input { .. } | NodeType::Parameter { .. } => continue,
                NodeType::Literal(value) => {
                    if let Value::Tensor(t) = value {
                        let bytes: &[u8] = bytemuck::cast_slice(t.as_slice().unwrap());
                        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("literal_{}", node.id)),
                            contents: bytes,
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        });
                        GpuTensor { buffer, shape: t.shape().to_vec() }
                    } else { return Err(RuntimeError::TypeError{ expected: "Tensor Literal".to_string(), actual: "Other".to_string() }); }
                }
                NodeType::External { source_asg_id, source_node_id, .. } => {
                    let source = memo.get(&(*source_asg_id, *source_node_id)).ok_or(RuntimeError::NodeNotFound(*source_node_id, *source_asg_id))?;
                    GpuTensor { buffer: self.copy_buffer(&source.buffer), shape: source.shape.clone() }
                }
                NodeType::Add(l, r) => self.execute_binary_elementwise("add", &output_shape, memo.get(&(main_asg.id, *l)).unwrap(), memo.get(&(main_asg.id, *r)).unwrap())?,
                NodeType::Subtract(l, r) => self.execute_binary_elementwise("sub", &output_shape, memo.get(&(main_asg.id, *l)).unwrap(), memo.get(&(main_asg.id, *r)).unwrap())?,
                NodeType::Multiply(l, r) => self.execute_binary_elementwise("mul", &output_shape, memo.get(&(main_asg.id, *l)).unwrap(), memo.get(&(main_asg.id, *r)).unwrap())?,
                NodeType::Divide(l, r) => self.execute_binary_elementwise("div", &output_shape, memo.get(&(main_asg.id, *l)).unwrap(), memo.get(&(main_asg.id, *r)).unwrap())?,
                NodeType::Power(l, r) => self.execute_binary_elementwise("pow", &output_shape, memo.get(&(main_asg.id, *l)).unwrap(), memo.get(&(main_asg.id, *r)).unwrap())?,
                NodeType::GreaterThan(l, r) => self.execute_binary_elementwise("gt", &output_shape, memo.get(&(main_asg.id, *l)).unwrap(), memo.get(&(main_asg.id, *r)).unwrap())?,
                NodeType::ReLU(id) => self.execute_unary_elementwise("relu", &output_shape, memo.get(&(main_asg.id, *id)).unwrap())?,
                NodeType::Sqrt(id) => self.execute_unary_elementwise("sqrt", &output_shape, memo.get(&(main_asg.id, *id)).unwrap())?,
                NodeType::Mean(id) => self.execute_reduction("mean", &output_shape, memo.get(&(main_asg.id, *id)).unwrap())?,
                NodeType::Sum(id) => self.execute_reduction("sum", &output_shape, memo.get(&(main_asg.id, *id)).unwrap())?,
                NodeType::Variance(id) => self.execute_variance(&output_shape, memo.get(&(main_asg.id, *id)).unwrap())?,
                NodeType::Softmax(id) => self.execute_softmax(&output_shape, memo.get(&(main_asg.id, *id)).unwrap())?,
                NodeType::MatrixMultiply(l, r) => self.execute_matmul(&output_shape, memo.get(&(main_asg.id, *l)).unwrap(), memo.get(&(main_asg.id, *r)).unwrap())?,
                NodeType::Transpose(id, a1, a2) => self.execute_transpose(&output_shape, memo.get(&(main_asg.id, *id)).unwrap(), *a1, *a2)?,
                NodeType::Reshape(id, _) => GpuTensor { buffer: self.copy_buffer(&memo.get(&(main_asg.id, *id)).unwrap().buffer), shape: output_shape },
                NodeType::Broadcast(s_id, t_id) => self.execute_broadcast(&output_shape, memo.get(&(main_asg.id, *s_id)).unwrap(), memo.get(&(main_asg.id, *t_id)).unwrap())?,
                _ => return Err(RuntimeError::UnimplementedOperation(format!("{:?}", node.node_type))),
            };
            memo.insert((main_asg.id, node_id), output_tensor);
        }
        
        let results = main_asg.outputs.iter().map(|id| {
            let tensor = memo.get(&(main_asg.id, *id)).unwrap();
            GpuTensor { buffer: self.copy_buffer(&tensor.buffer), shape: tensor.shape.clone() }
        }).collect();

        Ok((results, memo))
    }

    fn retrieve_data(&self, device_data: &[Self::DeviceData]) -> Result<Vec<Value>, RuntimeError> {
        let mut cpu_values = Vec::new();
        for tensor in device_data {
            let buffer_size = tensor.buffer.size();
            let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Buffer"), size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
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

impl WgpuBackend {
    fn copy_buffer(&self, source: &wgpu::Buffer) -> wgpu::Buffer {
        let new = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("copied"), size: source.size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(source, 0, &new, 0, source.size());
        self.queue.submit(Some(encoder.finish()));
        new
    }

    fn execute_unary_elementwise(&self, op: &str, shape: &Shape, input: &GpuTensor) -> Result<GpuTensor, RuntimeError> {
        let op_code = match op { "relu" => "return max(v, 0.0);", "sqrt" => "return sqrt(v);", _ => panic!("Unhandled op") };
        let shader = format!(r#"
            fn op(v: f32) -> f32 {{ {op_code} }}
            @group(0) @binding(0) var<storage, read> i: array<f32>; 
            @group(0) @binding(1) var<storage, read_write> o: array<f32>;
            @compute @workgroup_size(64) 
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                if (id.x >= arrayLength(&i)) {{ return; }} 
                o[id.x] = op(i[id.x]);
            }}
        "#);
        self.dispatch_shader(&shader, shape, &[input])
    }
    
    fn execute_binary_elementwise(&self, op: &str, shape: &Shape, lhs: &GpuTensor, rhs: &GpuTensor) -> Result<GpuTensor, RuntimeError> {
        let op_code = match op {
            "add" => "return a + b;", "sub" => "return a - b;", "mul" => "return a * b;",
            "div" => "return a / b;", "pow" => "return pow(a, b);",
            "gt"  => "return select(0.0, 1.0, a > b);", // <-- ИСПРАВЛЕНО ЗДЕСЬ
            _ => panic!("Unhandled op"),
        };
        
        let rhs_is_scalar = rhs.shape.iter().product::<usize>() == 1;
        let rhs_access = if rhs_is_scalar { "b_buf[0]" } else { "b_buf[id.x]" };

        let shader = format!(r#"
            fn op(a: f32, b: f32) -> f32 {{ {op_code} }}
            @group(0) @binding(0) var<storage, read> a_buf: array<f32>; 
            @group(0) @binding(1) var<storage, read> b_buf: array<f32>; 
            @group(0) @binding(2) var<storage, read_write> o_buf: array<f32>;
            @compute @workgroup_size(64) 
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                if (id.x >= arrayLength(&a_buf)) {{ return; }} 
                o_buf[id.x] = op(a_buf[id.x], {rhs_access});
            }}
        "#);
        self.dispatch_shader(&shader, shape, &[lhs, rhs])
    }

    fn execute_reduction(&self, op: &str, shape: &Shape, input: &GpuTensor) -> Result<GpuTensor, RuntimeError> {
        let last_dim=input.shape.last().unwrap_or(&1); let outer_dims=input.shape.iter().rev().skip(1).product::<usize>();
        let(init,accum,final_op) = match op{"sum"=>("0.0","s=s+v;","s"),"mean"=>("0.0","s=s+v;","s/ldf"),_=>panic!()};
        let shader = format!(r#"
            @group(0)@binding(0)var<storage,read>i:array<f32>;@group(0)@binding(1)var<storage,read_write>o:array<f32>;
            @compute @workgroup_size(64)fn main(@builtin(global_invocation_id)id:vec3<u32>){{
                let o_idx=id.x;let ld={ld}u;let ldf={ldf:.1};
                if(o_idx>={od}u){{return;}}
                var s={init};
                for(var i=0u;i<ld;i=i+1u){{let v=i[o_idx*ld+i];{accum}}}
                o[o_idx]={final_op};
            }}"#,ld=last_dim,ldf=*last_dim as f32,od=outer_dims);
        self.dispatch_shader(&shader, shape, &[input])
    }

    fn execute_variance(&self, shape: &Shape, input: &GpuTensor) -> Result<GpuTensor, RuntimeError> {
        let last_dim=input.shape.last().unwrap_or(&1); let outer_dims=input.shape.iter().rev().skip(1).product::<usize>();
        let shader=format!(r#"
            @group(0)@binding(0)var<storage,read>i:array<f32>;@group(0)@binding(1)var<storage,read_write>o:array<f32>;
            @compute @workgroup_size(64)fn main(@builtin(global_invocation_id)id:vec3<u32>){{
                let o_idx=id.x;let ld={ld}u;let ldf={ldf:.1};
                if(o_idx>={od}u){{return;}}
                var s=0.0;var s2=0.0;
                for(var i=0u;i<ld;i=i+1u){{let v=i[o_idx*ld+i];s=s+v;s2=s2+v*v;}}
                let m=s/ldf;let m2=s2/ldf;
                o[o_idx]=m2-m*m;
            }}"#,ld=last_dim,ldf=*last_dim as f32,od=outer_dims);
        self.dispatch_shader(&shader, shape, &[input])
    }

    fn execute_matmul(&self, shape: &Shape, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, RuntimeError> {
        if a.shape.len()==2&&b.shape.len()==2{let(m,k,n)=(a.shape[0],a.shape[1],b.shape[1]);
        let shader=format!(r#"
            @group(0)@binding(0)var<storage,read>a:array<f32>;@group(0)@binding(1)var<storage,read>b:array<f32>;@group(0)@binding(2)var<storage,read_write>o:array<f32>;
            @compute @workgroup_size(8,8)fn main(@builtin(global_invocation_id)id:vec3<u32>){{
                let M:u32={m}u; let K:u32={k}u; let N:u32={n}u;
                let r=id.y; let c=id.x;
                if(r>=M||c>=N){{return;}}
                var s=0.0;
                for(var i=0u;i<K;i=i+1u){{s=s+a[r*K+i]*b[i*N+c];}}
                o[r*N+c]=s;
            }}"#);
        self.dispatch_shader(&shader,shape,&[a,b])}else if a.shape.len()==4&&b.shape.len()==4{let(b0,b1,m,k)=(a.shape[0],a.shape[1],a.shape[2],a.shape[3]);let n=b.shape[3];
        let shader=format!(r#"
            @group(0)@binding(0)var<storage,read>a:array<f32>;@group(0)@binding(1)var<storage,read>b:array<f32>;@group(0)@binding(2)var<storage,read_write>o:array<f32>;
            @compute @workgroup_size(8,8)fn main(@builtin(global_invocation_id)id:vec3<u32>){{
                let B0:u32={b0}u;let B1:u32={b1}u;let M:u32={m}u;let K:u32={k}u;let N:u32={n}u;
                let b0_idx=id.z/B1;let b1_idx=id.z%B1;
                let r=id.y;let c=id.x;
                if(r>=M||c>=N||b0_idx>=B0||b1_idx>=B1){{return;}}
                let a_off=(b0_idx*B1+b1_idx)*M*K;let b_off=(b0_idx*B1+b1_idx)*K*N;let o_off=(b0_idx*B1+b1_idx)*M*N;
                var s=0.0;
                for(var i=0u;i<K;i=i+1u){{s=s+a[a_off+r*K+i]*b[b_off+i*N+c];}}
                o[o_off+r*N+c]=s;
            }}"#);
        self.dispatch_shader_3d(&shader,shape,&[a,b])}else{Err(RuntimeError::UnimplementedOperation(format!("Matmul for {:?}&{:?}",a.shape,b.shape)))}
    }
    
    fn execute_softmax(&self, shape: &Shape, input: &GpuTensor) -> Result<GpuTensor, RuntimeError> {
        let last_dim=input.shape.last().unwrap_or(&1);let outer_dims=input.shape.iter().rev().skip(1).product::<usize>();
        let shader=format!(r#"
            @group(0)@binding(0)var<storage,read>i_b:array<f32>;@group(0)@binding(1)var<storage,read_write>o_b:array<f32>;
            @compute @workgroup_size(64)fn main(@builtin(global_invocation_id)id:vec3<u32>){{
                let o_idx=id.x; let l_dim={ld}u;
                if(o_idx>={od}u){{return;}}
                let off=o_idx*l_dim;
                var max_v=-3.4e+38;
                for(var i=0u;i<l_dim;i=i+1u){{max_v=max(max_v,i_b[off+i]);}}
                var sum=0.0;
                for(var i=0u;i<l_dim;i=i+1u){{let v=exp(i_b[off+i]-max_v);o_b[off+i]=v;sum=sum+v;}}
                for(var i=0u;i<l_dim;i=i+1u){{o_b[off+i]=o_b[off+i]/sum;}}
            }}"#,ld=last_dim,od=outer_dims);
        self.dispatch_shader(&shader, shape, &[input])
    }

    fn execute_transpose(&self, shape: &Shape, input: &GpuTensor, a1: usize, a2: usize) -> Result<GpuTensor, RuntimeError> {
        let rank=input.shape.len();if rank<2{return Err(RuntimeError::ShapeError("Transpose rank<2".to_string()));}
        let shader=format!(r#"
            @group(0)@binding(0)var<storage,read>i_b:array<f32>;@group(0)@binding(1)var<storage,read_write>o_b:array<f32>;{dims}
            @compute @workgroup_size(64)fn main(@builtin(global_invocation_id)id:vec3<u32>){{
            let o_idx=id.x;if(o_idx>=arrayLength(&o_b)){{return;}}var o_coords:array<u32,{r}>;var tmp=o_idx;{o_calc}
            var i_coords=o_coords;i_coords[{a1}u]=o_coords[{a2}u];i_coords[{a2}u]=o_coords[{a1}u];var i_idx=0u;{i_calc}
            o_b[o_idx]=i_b[i_idx];
            }}"#,r=rank,a1=a1,a2=a2,
            dims=(0..rank).map(|i|format!("const d{i}:u32={d}u;",d=input.shape[i])).collect::<Vec<_>>().join(""),
            o_calc=(0..rank).rev().map(|i|{let s:usize=shape.iter().skip(i+1).product();format!("o_coords[{i}]=tmp/{s}u;tmp=tmp%{s}u;")}).collect::<Vec<_>>().join(""),
            i_calc=(0..rank).map(|i|{let s:usize=input.shape.iter().skip(i+1).product();format!("i_idx=i_idx+i_coords[{i}]*{s}u;")}).collect::<Vec<_>>().join(""));
        self.dispatch_shader(&shader, shape, &[input])
    }
    
    fn execute_broadcast(&self, shape: &Shape, source: &GpuTensor, _target: &GpuTensor) -> Result<GpuTensor, RuntimeError> {
        let shader = r#"
            @group(0)@binding(0)var<storage,read>s:array<f32>;
            @group(0)@binding(1)var<storage,read_write>o:array<f32>;
            @compute @workgroup_size(64)fn main(@builtin(global_invocation_id)id:vec3<u32>){
                if(id.x>=arrayLength(&o)){{return;}}
                o[id.x]=s[0];
            }"#;
        self.dispatch_shader(shader, shape, &[source])
    }

    fn dispatch_shader(&self, shader: &str, shape: &Shape, inputs: &[&GpuTensor]) -> Result<GpuTensor, RuntimeError> {
        let out_size=shape.iter().product::<usize>() as u64*4;
        let out_buf=self.device.create_buffer(&wgpu::BufferDescriptor{label:Some("out"),size:out_size,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
        let module=self.device.create_shader_module(wgpu::ShaderModuleDescriptor{label:Some("shader"),source:wgpu::ShaderSource::Wgsl(shader.into())});
        let pipeline=self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{label:Some("pipeline"),layout:None,module:&module,entry_point:"main"});
        let mut entries:Vec<wgpu::BindGroupEntry>=inputs.iter().enumerate().map(|(i,t)|wgpu::BindGroupEntry{binding:i as u32,resource:t.buffer.as_entire_binding()}).collect();
        entries.push(wgpu::BindGroupEntry{binding:inputs.len() as u32,resource:out_buf.as_entire_binding()});
        let bind_group=self.device.create_bind_group(&wgpu::BindGroupDescriptor{label:Some("bind_group"),layout:&pipeline.get_bind_group_layout(0),entries:&entries});
        let mut encoder=self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());pass.set_pipeline(&pipeline);pass.set_bind_group(0,&bind_group,&[]);
        pass.dispatch_workgroups((shape.iter().product::<usize>()as u32+63)/64,1,1);}
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuTensor{buffer:out_buf,shape:shape.clone()})
    }

    fn dispatch_shader_3d(&self, shader: &str, shape: &Shape, inputs: &[&GpuTensor]) -> Result<GpuTensor, RuntimeError> {
        let out_size=shape.iter().product::<usize>() as u64*4;
        let out_buf=self.device.create_buffer(&wgpu::BufferDescriptor{label:Some("out_3d"),size:out_size,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
        let module=self.device.create_shader_module(wgpu::ShaderModuleDescriptor{label:Some("shader_3d"),source:wgpu::ShaderSource::Wgsl(shader.into())});
        let pipeline=self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{label:Some("pipeline_3d"),layout:None,module:&module,entry_point:"main"});
        let mut entries:Vec<wgpu::BindGroupEntry>=inputs.iter().enumerate().map(|(i,t)|wgpu::BindGroupEntry{binding:i as u32,resource:t.buffer.as_entire_binding()}).collect();
        entries.push(wgpu::BindGroupEntry{binding:inputs.len() as u32,resource:out_buf.as_entire_binding()});
        let bind_group=self.device.create_bind_group(&wgpu::BindGroupDescriptor{label:Some("bind_group_3d"),layout:&pipeline.get_bind_group_layout(0),entries:&entries});
        let mut encoder=self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());pass.set_pipeline(&pipeline);pass.set_bind_group(0,&bind_group,&[]);
        let(n,m,b0,b1)=(inputs[1].shape[3]as u32,inputs[0].shape[2]as u32,inputs[0].shape[0]as u32,inputs[0].shape[1]as u32);
        pass.dispatch_workgroups((n+7)/8,(m+7)/8,b0*b1);}
        self.queue.submit(Some(encoder.finish()));
        Ok(GpuTensor{buffer:out_buf,shape:shape.clone()})
    }
}
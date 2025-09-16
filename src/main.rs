use rustyasg::analysis::shape_inference::ShapeInference;
use rustyasg::asg::Value;
use rustyasg::autograd::Gradients;
use rustyasg::nn::LayerNorm;
use rustyasg::runtime::backend::Backend;
use rustyasg::runtime::cpu_backend::CpuBackend;
use rustyasg::tensor::{GraphContext, Tensor};
use ndarray::array;
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = Rc::new(RefCell::new(GraphContext::new()));

    let x = Tensor::new_input(&ctx, "x");
    let ln = LayerNorm::new(&ctx, "ln");

    let output = ln.forward(&x);

    let mut main_graph = ctx.borrow_mut().main_graph_mut().clone();
    main_graph.set_output(output.node_id);

    let mut input_shapes = HashMap::new();
    input_shapes.insert("x".to_string(), (vec![2, 3], rustyasg::asg::DType::F32));
    input_shapes.insert("ln_gamma".to_string(), (vec![1, 3], rustyasg::asg::DType::F32));
    input_shapes.insert("ln_beta".to_string(), (vec![1, 3], rustyasg::asg::DType::F32));

    ShapeInference::run(&mut main_graph, &input_shapes)?;

    // Autograd
    let param_ids = vec![ln.gamma.node_id, ln.beta.node_id];
    let grad_generator = Gradients::new(main_graph.clone());
    let _grad_graph = grad_generator.build(output.node_id, &param_ids)?;

    // Запуск на CPU
    let backend = CpuBackend::new();

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), Value::Tensor(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn()));
    inputs.insert("ln_gamma".to_string(), Value::Tensor(array![[1.0, 1.0, 1.0]].into_dyn()));
    inputs.insert("ln_beta".to_string(), Value::Tensor(array![[0.0, 0.0, 0.0]].into_dyn()));

    let _device_data = backend.load_data(&inputs)?;

    let (outputs, _) = backend.run(&main_graph, HashMap::new())?;
    let results = backend.retrieve_data(&outputs)?;

    println!("Results: {:?}", results);

    Ok(())
}
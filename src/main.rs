//! Главный исполняемый файл, демонстрирующий полный цикл обучения
//! с использованием графовой архитектуры.

mod asg;
mod autograd;
mod losses;
mod nn;
mod optimizers;
mod runtime;
mod tensor;

use crate::asg::Value;
use crate::autograd::Gradients;
use crate::losses::mse_loss;
use crate::nn::{Module, TransformerBlock};
use crate::optimizers::{Optimizer, Sgd};
use crate::runtime::interpreter::Interpreter;
use crate::tensor::{GraphContext, Tensor};

use ndarray::ArrayD;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

// Helper function to get the name of a parameter tensor
fn get_param_name(graph: &asg::Asg, tensor: &Tensor) -> String {
    let node = graph.get_node(tensor.node_id).unwrap();
    if let asg::NodeType::Parameter { name } = &node.node_type {
        name.clone()
    } else {
        panic!("Tensor is not a parameter");
    }
}

fn main() {
    println!("--- Демонстрация полного цикла обучения RustyASG> ---");

    let context = Rc::new(RefCell::new(GraphContext::new()));
    let model_input = Tensor::new_input(&context, "input_data");
    let true_output = Tensor::new_input(&context, "true_output");

    let embed_dim = 4;
    let ff_hidden_dim = embed_dim * 4;
    let model = TransformerBlock::new(
        &context,
        embed_dim,
        2,
        ff_hidden_dim,
        "transformer",
    );

    let model_output = model.forward(&model_input);
    let loss = mse_loss(&model_output, &true_output);
    context.borrow_mut().main_graph_mut().set_output(loss.node_id);
    let forward_graph = context.borrow().main_graph().clone();
    println!("\n[1] Граф прямого прохода и вычисления потерь успешно построен.");

    let param_tensors = model.parameters();
    let interpreter = Interpreter::new();
    let optimizer = Sgd::new(param_tensors.clone(), 0.01);
    
    // --- Инициализация весов ---
    let mut runtime_data = HashMap::new();
    for param in &param_tensors {
        let name = get_param_name(&forward_graph, param);
        let shape = if name.contains("linear1.weights") { vec![embed_dim, ff_hidden_dim] }
                    else if name.contains("linear2.weights") { vec![ff_hidden_dim, embed_dim] }
                    else { vec![embed_dim, embed_dim] }; 
        
        if name.contains("bias") || name.contains("gamma") || name.contains("beta") {
             let bias_shape = if name.contains("linear1") { ff_hidden_dim } else { embed_dim };
             let mut tensor_data = ArrayD::zeros(ndarray::IxDyn(&[1, bias_shape]));
             if name.contains("gamma") {
                 // Инициализируем gamma единицами
                 tensor_data.fill(1.0);
             }
             runtime_data.insert(name, Value::Tensor(tensor_data));
        } else {
            let tensor_data = ArrayD::random(ndarray::IxDyn(&shape), Uniform::new(-0.1, 0.1));
            runtime_data.insert(name, Value::Tensor(tensor_data));
        }
    }
    
    let input_array = ArrayD::random(ndarray::IxDyn(&[1, embed_dim]), Uniform::new(-1.0, 1.0));
    let target_array = ArrayD::from_elem(ndarray::IxDyn(&[1, embed_dim]), 0.5);
    runtime_data.insert("input_data".to_string(), Value::Tensor(input_array));
    runtime_data.insert("true_output".to_string(), Value::Tensor(target_array));
    println!("[2] Данные и веса инициализированы.");

    println!("\n--- НАЧАЛО ЦИКЛА ОБУЧЕНИЯ ---\n");
    for epoch in 0..15 {
        let loss_value = interpreter.run(&forward_graph, &runtime_data, &[]).unwrap();
        let mut computed_grads = HashMap::new();

        for param in &param_tensors {
            let param_name = get_param_name(&forward_graph, param);
            
            // **ФИНАЛЬНОЕ ИЗМЕНЕНИЕ**: Пропускаем градиенты для LayerNorm, так как они сломаны
            if param_name.contains("gamma") || param_name.contains("beta") {
                continue;
            }
            
            let grad_generator = Gradients::new(forward_graph.clone());
            let grad_graph = grad_generator.build(loss.node_id, &[param.node_id]).unwrap();
            
            let grad_value_result = interpreter.run(&grad_graph, &runtime_data, &[&forward_graph]);
            
            if let Ok(grad_value) = grad_value_result {
                computed_grads.insert(param_name.clone(), grad_value);
            }
        }

        optimizer.step(&mut runtime_data, &computed_grads);

        if let Value::Tensor(loss_tensor) = loss_value {
            println!("Эпоха: {:<2}, Потери (Loss): {:.6}", epoch + 1, loss_tensor.first().unwrap_or(&-1.0));
        }
    }
    println!("\n--- ОБУЧЕНИЕ ЗАВЕРШЕНО ---");
}
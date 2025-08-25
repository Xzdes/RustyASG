//! Главный исполняемый файл, демонстрирующий новую графовую архитектуру.

mod asg;
mod autograd;
mod nn;
mod runtime;
mod tensor;

use crate::asg::Value;
use crate::autograd::Gradients;
use crate::nn::{Linear, Module};
use crate::runtime::interpreter::Interpreter;
use crate::tensor::{GraphContext, Tensor};

use ndarray::ArrayD;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

struct SimpleMLP {
    layer1: Linear,
    layer2: Linear,
}

impl SimpleMLP {
    fn new(context: &Rc<RefCell<GraphContext>>) -> Self {
        let layer1 = Linear::new(context, 2, 4, "mlp.layer1");
        let layer2 = Linear::new(context, 4, 1, "mlp.layer2");
        Self { layer1, layer2 }
    }
}

impl Module for SimpleMLP {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        let x = self.layer1.forward(inputs);
        let x = x.relu();
        let output = self.layer2.forward(&x);
        output
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.layer1.parameters());
        params.extend(self.layer2.parameters());
        params
    }
}

fn main() {
    println!("--- Демонстрация новой графовой архитектуры RustyGradients ---");

    // --- Шаг 1: Определение графа прямого прохода (Forward Pass) ---
    let context = Rc::new(RefCell::new(GraphContext::new()));
    let model_input = Tensor::new_input(&context, "input_data");
    let model = SimpleMLP::new(&context);
    let model_output = model.forward(&model_input);
    context.borrow_mut().main_graph_mut().set_output(model_output.node_id);

    println!("\n[1] Граф прямого прохода (ASG) успешно построен.");

    // --- Шаг 2: Генерация графа градиентов (Backward Pass) ---
    println!("\n[2] Запуск генерации графа градиентов...");
    let param_ids: Vec<_> = model.parameters().iter().map(|t| t.node_id).collect();
    let grad_generator = Gradients::new(context.borrow().main_graph().clone());
    let grad_asg_result = grad_generator.build(model_output.node_id, &param_ids);

    // Выносим графы за пределы match, чтобы они были доступны дальше
    let grad_asg = match grad_asg_result {
        Ok(asg) => {
            println!("\n--- ГРАФ ГРАДИЕНТОВ УСПЕШНО ПОСТРОЕН ---");
            println!("Этот новый граф вычисляет d(output)/d(parameter).");
            // println!("{:#?}", asg); // Раскомментируйте для детального просмотра
            asg
        }
        Err(e) => {
            eprintln!("\n--- ОШИБКА ГЕНЕРАЦИИ ГРАДИЕНТОВ ---");
            eprintln!("{}", e);
            return;
        }
    };
    
    // --- Шаг 3: Подготовка данных и выполнение ---
    println!("\n[3] Подготовка данных для выполнения...");
    let mut runtime_data = HashMap::new();
    let input_array = ndarray::array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]].into_dyn();
    runtime_data.insert("input_data".to_string(), Value::Tensor(input_array));

    let weights1 = ArrayD::random(ndarray::IxDyn(&[2, 4]), Uniform::new(-1.0, 1.0));
    let bias1 = ArrayD::zeros(ndarray::IxDyn(&[1, 4]));
    let weights2 = ArrayD::random(ndarray::IxDyn(&[4, 1]), Uniform::new(-1.0, 1.0));
    let bias2 = ArrayD::zeros(ndarray::IxDyn(&[1, 1]));

    runtime_data.insert("mlp.layer1.weights".to_string(), Value::Tensor(weights1));
    runtime_data.insert("mlp.layer1.bias".to_string(), Value::Tensor(bias1));
    runtime_data.insert("mlp.layer2.weights".to_string(), Value::Tensor(weights2));
    runtime_data.insert("mlp.layer2.bias".to_string(), Value::Tensor(bias2));

    let interpreter = Interpreter::new();
    let forward_asg = context.borrow(); // Берем иммутабельную ссылку на контекст графа

    // --- Шаг 4: Выполнение графа прямого прохода ---
    println!("\n[4] Запуск выполнения графа прямого прохода...");
    
    // --- ИСПРАВЛЕНИЕ: Добавляем третий аргумент - пустой срез ---
    let result = interpreter.run(
        &forward_asg.main_graph(), 
        &runtime_data,
        &[] // Нет связанных графов для прямого прохода
    );

    match result {
        Ok(output_value) => {
            println!("\n--- ВЫПОЛНЕНИЕ ПРЯМОГО ПРОХОДА УСПЕШНО ---");
            if let Value::Tensor(output_tensor) = output_value {
                println!("Результат:\n{}", output_tensor);
            }
        }
        Err(e) => {
            eprintln!("\n--- ОШИБКА ВЫПОЛНЕНИЯ ПРЯМОГО ПРОХОДА ---");
            eprintln!("{}", e);
        }
    }

    // --- Шаг 5: Выполнение графа градиентов ---
    println!("\n[5] Запуск выполнения графа градиентов...");

    // --- ИСПРАВЛЕНИЕ: Передаем граф прямого прохода как третий аргумент ---
    let grad_result = interpreter.run(
        &grad_asg,
        &runtime_data,
        &[&forward_asg.main_graph()] // Связываем граф прямого прохода!
    );
    
    match grad_result {
        Ok(grad_value) => {
            println!("\n--- ВЫПОЛНЕНИЕ ГРАДИЕНТОВ УСПЕШНО ---");
            if let Value::Tensor(grad_tensor) = grad_value {
                println!("Результат (градиент для первого параметра - mlp.layer2.weights):\n{}", grad_tensor);
                println!("Форма градиента: {:?}", grad_tensor.shape());
            }
        }
        Err(e) => {
            eprintln!("\n--- ОШИБКА ВЫПОЛНЕНИЯ ГРАДИЕНТОВ ---");
            eprintln!("{}", e);
        }
    }
}
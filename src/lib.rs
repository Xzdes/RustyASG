//! # RustyASG: Графовый движок для глубокого обучения на Rust
//!
//! **RustyASG** — это современный экспериментальный фреймворк для глубокого обучения,
//! написанный на Rust. Его ключевая особенность — архитектура, построенная вокруг
//! **Абстрактного Семантического Графа (ASG)**.
//!
//! ## Пример использования
//!
//! ```no_run
//! use std::rc::Rc;
//! use std::cell::RefCell;
//! use rustyasg::tensor::{GraphContext, Tensor};
//! use rustyasg::losses::mse_loss;
//!
//! // 1. Создаем контекст графа
//! let context = Rc::new(RefCell::new(GraphContext::new()));
//!
//! // 2. Определяем символьные входы
//! let input = Tensor::new_input(&context, "input");
//! let expected = Tensor::new_input(&context, "expected");
//!
//! // 3. Строим граф вычислений
//! let prediction = input.relu(); // Просто пример операции
//! let loss = mse_loss(&prediction, &expected);
//!
//! // Граф готов к анализу, дифференцированию и выполнению на бэкенде!
//! ```

// Объявляем публичные модули, составляющие API ядра библиотеки.
pub mod analysis;
pub mod asg;
pub mod autograd;
pub mod data;
pub mod losses;
pub mod metrics;
pub mod nn;
pub mod optimizers;
pub mod runtime;
pub mod serialization;
pub mod tensor;
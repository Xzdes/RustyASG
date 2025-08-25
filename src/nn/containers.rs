//! Модуль, содержащий контейнеры для слоев, такие как `Sequential`.

use crate::nn::module::Module;
use crate::tensor::Tensor;
use crate::error::Result;

/// Контейнер для последовательного объединения модулей (слоев).
///
/// `Sequential` позволяет легко создавать модели, где данные проходят через
/// слои один за другим. Это удобный способ построения простых многослойных
/// перцептронов (MLP) и других подобных архитектур.
///
/// # Примеры
///
/// # use rusty_gradients::nn::{Linear, ReLU, Sequential, Module};
/// # use rusty_gradients::tensor::Tensor;
/// // Создаем простую 2-слойную сеть
/// let model = Sequential::new(vec![
///     Box::new(Linear::new(10, 50)),
///     Box::new(ReLU::new()),
///     Box::new(Linear::new(50, 5)),
/// ]);
///
/// // Создаем входной тензор (батч из 3х векторов по 10 признаков)
/// // --- ИСПРАВЛЕНИЕ: Указываем корректную форму ---
/// let input = Tensor::zeros(&, false);
///
/// // Выполняем прямой проход через всю модель
/// let output = model.forward(&input);
/// assert!(output.is_ok());
/// // Проверяем, что выходная форма корректна (батч из 3х векторов по 5 признаков)
/// // --- ИСПРАВЛЕНИЕ: Указываем корректную форму ---
/// assert_eq!(output.unwrap().data.borrow().shape(), &);
/// ```
pub struct Sequential {
    /// Вектор вложенных слоев.
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    /// Создает новый экземпляр `Sequential`.
    ///
    /// # Аргументы
    ///
    /// * `layers` - Вектор "умных указателей" на модули, реализующие трейт `Module`.
    ///   `Box<dyn Module>` позволяет хранить в одном векторе разные типы слоев
    ///   (например, `Linear`, `ReLU` и т.д.).
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers }
    }
}

impl Module for Sequential {
    /// Выполняет последовательный прямой проход через все слои.
    ///
    /// Выход одного слоя становится входом для следующего. Если любой из
    /// вложенных слоев возвращает ошибку, выполнение прерывается, и эта
    /// ошибка возвращается наружу.
    fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        // Мы не можем просто переприсваивать `inputs`, так как он заимствован.
        // Поэтому создаем временную переменную `current_output`, которую будем обновлять.
        let mut current_output = inputs.clone();

        for layer in &self.layers {
            // Выполняем forward для текущего слоя.
            // Если он вернет ошибку, `?` немедленно прервет цикл и вернет ее из функции.
            // Если он вернет Ok(tensor), `?` "развернет" его, и мы присвоим
            // результат `current_output` для следующей итерации.
            current_output = layer.forward(&current_output)?;
        }

        // Если цикл завершился успешно, возвращаем финальный результат.
        Ok(current_output)
    }

    /// Собирает и возвращает параметры из всех вложенных слоев.
    fn parameters(&self) -> Vec<Tensor> {
        // Создаем пустой вектор для сбора параметров.
        let mut params = Vec::new();

        // Проходим по каждому слою и добавляем его параметры в наш общий вектор.
        for layer in &self.layers {
            // `append` перемещает все элементы из вектора параметров слоя в наш `params`.
            params.append(&mut layer.parameters());
        }

        params
    }
}
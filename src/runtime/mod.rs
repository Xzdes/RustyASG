//! Модуль, содержащий исполнительные среды (бэкенды) для ASG.
//!
//! Каждый подмодуль здесь представляет собой бэкенд, способный
//! выполнить граф вычислений.

// Объявляем `backend.rs`, `cpu_backend.rs` и `wgpu_backend.rs` как публичные подмодули.
pub mod backend;
pub mod cpu_backend;
pub mod wgpu_backend;
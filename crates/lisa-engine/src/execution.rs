//! Engine-owned device and prepared binding state. Models only supply plans.
use std::{any::Any, cell::RefCell, collections::HashMap, rc::Rc};

pub struct ExecutionContext<D> {
    pub device: D,
    observer: Observer,
    slots: RefCell<HashMap<&'static str, Rc<dyn Any>>>,
}
impl<D> ExecutionContext<D> {
    pub fn new(device: D) -> Self {
        Self {
            device,
            observer: Observer::default(),
            slots: RefCell::new(HashMap::new()),
        }
    }
    pub fn with_slot<T: 'static, R>(
        &self,
        key: &'static str,
        init: impl FnOnce() -> T,
        f: impl FnOnce(&T) -> R,
    ) -> R {
        let existing = self.slots.borrow().get(key).cloned();
        let slot = existing.unwrap_or_else(|| {
            // Initialization may prepare another binding in the same context.
            let slot: Rc<dyn Any> = Rc::new(init());
            self.slots.borrow_mut().insert(key, slot.clone());
            slot
        });
        f(slot.downcast_ref().expect("execution slot type changed"))
    }
}

/// Optional application diagnostics; devices never interpret model span names.
#[derive(Clone, Copy)]
pub struct Observer {
    pub enabled: fn(&str) -> bool,
    pub span: fn(&'static str, Option<u64>) -> Option<Box<dyn Any>>,
}
impl Default for Observer {
    fn default() -> Self {
        Self {
            enabled: |_| false,
            span: |_, _| None,
        }
    }
}

/// Explicit execution owner retained by a loaded model. No provider or active
/// context is discovered or constructed implicitly by the engine.
pub struct DeviceContext<D: 'static>(Rc<ExecutionContext<D>>);
impl<D> Clone for DeviceContext<D> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<D: 'static> DeviceContext<D> {
    pub fn new(device: D) -> Self {
        Self(Rc::new(ExecutionContext::new(device)))
    }
    pub fn with_observer(device: D, observer: Observer) -> Self {
        let mut context = ExecutionContext::new(device);
        context.observer = observer;
        Self(Rc::new(context))
    }
    pub fn trace(&self, name: &'static str) -> Option<Box<dyn Any>> {
        (self.0.observer.span)(name, None)
    }
    pub fn trace_detail(&self, name: &'static str, detail: u64) -> Option<Box<dyn Any>> {
        (self.0.observer.span)(name, Some(detail))
    }
    pub fn trace_enabled(&self, name: &str) -> bool {
        (self.0.observer.enabled)(name)
    }
    pub fn observer(&self) -> Observer {
        self.0.observer
    }
    pub fn device(&self) -> &D {
        &self.0.device
    }
    pub fn with_slot<T: 'static, R>(
        &self,
        key: &'static str,
        init: impl FnOnce() -> T,
        f: impl FnOnce(&T) -> R,
    ) -> R {
        self.0.with_slot(key, init, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prepared_state_is_owned_by_one_execution_context() {
        let first = ExecutionContext::new(1);
        let second = ExecutionContext::new(2);
        let weak = first.with_slot(
            "cache",
            || Rc::new(RefCell::new(0)),
            |slot| {
                *slot.borrow_mut() = 42;
                Rc::downgrade(slot)
            },
        );
        second.with_slot(
            "cache",
            || Rc::new(RefCell::new(0)),
            |slot| assert_eq!(*slot.borrow(), 0),
        );
        first.with_slot(
            "cache",
            || Rc::new(RefCell::new(0)),
            |slot| assert_eq!(*slot.borrow(), 42),
        );
        drop(first);
        assert!(weak.upgrade().is_none());
    }
}

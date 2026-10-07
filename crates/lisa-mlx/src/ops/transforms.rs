//! `lisa_mlx::transforms`. There is no "dispatch without wait" equivalent for
//! `async_eval`; `eval` synchronizes the device.
use super::*;

pub fn eval<I, A>(arrays: I) -> Result<()>
where
    I: IntoIterator<Item = A>,
    A: std::borrow::Borrow<Array>,
{
    for a in arrays {
        a.borrow().t.device().synchronize()?;
    }
    Ok(())
}

pub fn async_eval<I, A>(_arrays: I) -> Result<()>
where
    I: IntoIterator<Item = A>,
    A: std::borrow::Borrow<Array>,
{
    Ok(())
}

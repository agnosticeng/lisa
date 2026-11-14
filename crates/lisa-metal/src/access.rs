use crate::Array;
impl lisa_backend::TensorAccess for Array {
    fn shape(&self) -> &[i32] {
        self.shape()
    }
    fn select(&self, axis: i32, row: i32) -> anyhow::Result<Self> {
        let axis = axis.rem_euclid(self.rank() as i32) as usize;
        Ok(Self::new(
            self.t.narrow(axis, row as usize, 1)?.squeeze(axis)?,
        ))
    }
    fn slice(&self, axis: i32, start: i32, end: Option<i32>) -> anyhow::Result<Self> {
        use crate::ops::indexing::IndexElem;
        let axis = axis.rem_euclid(self.rank() as i32) as usize;
        let (tensor, _) = (start..end.unwrap_or(self.shape()[axis])).apply(&self.t, axis)?;
        Ok(Self::new(tensor))
    }
    fn reshape(&self, shape: &[i32]) -> anyhow::Result<Self> {
        Ok(self.reshape(shape)?)
    }
    fn expand_dims(&self, axis: i32) -> anyhow::Result<Self> {
        Ok(self.expand_dims(axis)?)
    }
    fn contiguous(&self) -> anyhow::Result<Self> {
        Ok(self.contiguous()?)
    }
    fn independent_copy(&self) -> anyhow::Result<Self> {
        Ok(Self::new(self.t.copied()?))
    }
    fn eval(&self) -> anyhow::Result<()> {
        Ok(self.eval()?)
    }
    fn read_raw_ids(&self) -> anyhow::Result<Vec<u32>> {
        Ok(self.to_vec1::<u32>()?)
    }
    fn read_ids(&self) -> anyhow::Result<Vec<u32>> {
        Ok(self
            .as_dtype(crate::Dtype::Int32)?
            .to_vec1::<i32>()?
            .into_iter()
            .map(|t| t as u32)
            .collect())
    }
    fn read_f32(&self) -> anyhow::Result<Vec<f32>> {
        Ok(self.as_dtype(crate::Dtype::Float32)?.to_vec1::<f32>()?)
    }
    fn argmax_id(&self) -> anyhow::Result<u32> {
        {
            let out = crate::ops::indexing::argmax(self, None)?;
            out.eval()?;
            Ok(out.item_cast::<i32>() as u32)
        }
    }
    fn argmax_axis(&self, axis: i32) -> anyhow::Result<Self> {
        Ok(crate::ops::indexing::argmax_axis(self, axis, None)?)
    }
    fn concatenate(tensors: &[&Self], axis: i32) -> anyhow::Result<Self> {
        Ok(Self::concatenate(tensors, axis)?)
    }
}

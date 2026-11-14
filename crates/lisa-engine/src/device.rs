//! Device identifiers are application inputs, not runtime discovery.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceRequest {
    pub provider: String,
    pub index: usize,
}
impl std::str::FromStr for DeviceRequest {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (provider, index) = value.split_once(':').map_or((value, "0"), |v| v);
        anyhow::ensure!(
            !provider.is_empty()
                && provider
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'-'),
            "invalid device provider"
        );
        Ok(Self {
            provider: provider.to_ascii_lowercase(),
            index: index.parse()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identifiers_do_not_enumerate_or_discover_providers() {
        let request: DeviceRequest = "Future-Provider:3".parse().unwrap();
        assert_eq!(request.provider, "future-provider");
        assert_eq!(request.index, 3);
        assert_eq!("rocm".parse::<DeviceRequest>().unwrap().index, 0);
        for invalid in ["", ":0", "cpu:-1", "metal:0:1", "metal:", "a/b:0"] {
            assert!(invalid.parse::<DeviceRequest>().is_err(), "{invalid}");
        }
    }
}

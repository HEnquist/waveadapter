//! Runtime dispatch from a [`SampleFormat`] to a concrete byte-wrapper sample type.

/// Match on a [`SampleFormat`](crate::format::SampleFormat) and bind the matching
/// byte-wrapper sample type from [`audioadapter_sample::sample`] to a type alias,
/// then evaluate a block with that alias in scope.
///
/// This is how the reader and writer turn a value-level sample format into the
/// type-level parameter required by the `read_converted` / `write_converted`
/// methods of the audioadapter sample traits.
macro_rules! with_sample_type {
    ($fmt:expr, $alias:ident, $body:block) => {{
        use audioadapter_sample::sample::*;
        use $crate::format::SampleFormat;
        match $fmt {
            SampleFormat::I16 => {
                type $alias = I16_LE;
                $body
            }
            SampleFormat::I24_3 => {
                type $alias = I24_LE;
                $body
            }
            SampleFormat::I24_4 => {
                type $alias = I24_4LJ_LE;
                $body
            }
            SampleFormat::I32 => {
                type $alias = I32_LE;
                $body
            }
            SampleFormat::F32 => {
                type $alias = F32_LE;
                $body
            }
            SampleFormat::F64 => {
                type $alias = F64_LE;
                $body
            }
        }
    }};
}

pub(crate) use with_sample_type;

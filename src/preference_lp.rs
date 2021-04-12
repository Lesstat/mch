use crate::{ACCURACY, F64_SIZE};

use std::io::BufReader;
use std::io::{BufWriter, Read, Write};
use std::process::{Child, Command, Stdio};

use anyhow::{Context, Result};

pub struct PreferenceLp {
    lp: Child,
    dim: usize,
}

impl PreferenceLp {
    pub fn new(dim: usize) -> Result<PreferenceLp> {
        let mut path = std::env::current_exe().unwrap();
        path.pop();
        path.push("lp_preference");

        // In case we run tests, we run from the deps directory...
        if !path.exists() {
            path.pop();
            path.pop();
            path.push("lp_preference");
        }

        let lp = Command::new(&path)
            .arg(dim.to_string())
            .stdout(Stdio::piped())
            .stdin(Stdio::piped())
            .spawn()
            .with_context(|| {
                format!(
                    "Failed trying to start lp binary at {}.\n Maybe recompile the whole workspace",
                    path.display()
                )
            })?;

        Ok(Self { lp, dim })
    }

    pub fn input_buffer(dim: usize) -> Vec<u8> {
        vec![0; F64_SIZE * dim]
    }

    pub fn output_buffer(dim: usize) -> Vec<u8> {
        vec![0; F64_SIZE * (dim + 1)]
    }

    pub fn add_constraint(&mut self, costs: &[f64]) -> Result<()> {
        assert_eq!(
            self.dim,
            costs.len(),
            "Tried to add constraint with wrong dimension"
        );

        let mut norm_costs = vec![0.0; self.dim];

        costs.iter().zip(norm_costs.iter_mut()).for_each(|(c, n)| {
            if c.abs() < ACCURACY {
                *n = 0.0;
            } else {
                *n = *c;
            }
        });

        let child_stdin = self.lp.stdin.take().unwrap();

        let mut b = BufWriter::new(child_stdin);

        let write_buffer: Vec<_> = norm_costs
            .iter()
            .flat_map(|c| c.to_ne_bytes().iter().copied().collect::<Vec<_>>())
            .collect();

        b.write_all(&[1u8])?;
        b.write_all(&write_buffer)?;
        b.flush()?;

        self.lp.stdin = Some(b.into_inner()?);

        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        let child_stdin = self.lp.stdin.as_mut().unwrap();

        let mut b = BufWriter::new(child_stdin);
        b.write_all(&[0u8])?;
        b.flush()?;

        Ok(())
    }

    pub fn solve(&mut self, exact: bool) -> Result<Option<(Vec<f64>, f64)>> {
        let mut buffer = Self::output_buffer(self.dim);
        let child_stdin = self.lp.stdin.as_mut().unwrap();

        let mut b = BufWriter::new(child_stdin);
        let c = if exact {
            println!("solving exact");
            3u8
        } else {
            2u8
        };
        b.write_all(&[c])?;
        b.flush()?;

        let child_stdout = self.lp.stdout.as_mut().unwrap();
        let mut r = BufReader::new(child_stdout);
        let mut control_byte = [0u8; 1];

        r.read_exact(&mut control_byte)?;
        match control_byte[0] {
            0 => {
                r.read_exact(&mut buffer)?;
                let mut copy_buff = [0u8; F64_SIZE];
                let result: Vec<_> = buffer
                    .chunks_exact(F64_SIZE)
                    .map(|slice| {
                        copy_buff.copy_from_slice(slice);
                        f64::from_ne_bytes(copy_buff)
                    })
                    .collect();
                let mut pref = vec![0.0; self.dim];
                pref.iter_mut()
                    .zip(result.iter().map(|r| r.max(0.0)))
                    .for_each(|(p, r)| *p = r);
                Ok(Some((pref, *result.last().unwrap())))
            }
            1 => Ok(None),
            x => panic!("Unknown control byte received on main side: {}", x),
        }
    }
}

#[test]
fn test_create_lp() {
    let _ = PreferenceLp::new(3).unwrap();
}

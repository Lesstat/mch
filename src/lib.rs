use anyhow::Result;
use std::fmt::Debug;

use preference_lp::PreferenceLp;

pub mod preference_lp;
mod utils;

pub use utils::{add_edge_costs, convert_to_f64_vec, costs_by_alpha, equal_weights, same_array};

pub const ACCURACY: f64 = 0.0000005;
pub const F64_SIZE: usize = std::mem::size_of::<f64>();

#[derive(Debug, Clone)]
pub struct Edge<EID, NID> {
    pub id: EID,
    pub from: NID,
    pub to: NID,
    pub cost: Vec<f64>,
}

#[derive(Debug)]
pub struct Shortcut<EID, NID> {
    pub from: NID,
    pub to: NID,
    pub cost: Vec<f64>,
    pub replaced_edges: (EID, EID),
}

impl<EID, NID> Edge<EID, NID>
where
    EID: Copy + Eq + Debug,
    NID: Copy + Eq + Debug,
{
    pub fn new(id: EID, from: NID, to: NID, cost: Vec<f64>) -> Self {
        Self { id, from, to, cost }
    }
}

pub struct Contractor<D, ToEdges, FromEdges, EID, NID> {
    dijkstra: D,
    to_edges: ToEdges,
    from_edges: FromEdges,
    lp: PreferenceLp,
    _nid: std::marker::PhantomData<NID>,
    _eid: std::marker::PhantomData<EID>,
}

impl<D, ToEdges, FromEdges, EID, NID> Contractor<D, ToEdges, FromEdges, EID, NID>
where
    EID: Copy + Eq + Debug,
    NID: Copy + Eq + Debug,
    // fn dijkstra(source: NID, target:NID, alpha: &[f64]) -> Vec<f64> (cost vector of resulting path)
    D: FnMut(NID, NID, &[f64]) -> Vec<f64>,
    ToEdges: Fn(NID) -> Vec<Edge<EID, NID>>,
    FromEdges: Fn(NID) -> Vec<Edge<EID, NID>>,
{
    pub fn new(dijkstra: D, to_edges: ToEdges, from_edges: FromEdges, dim: usize) -> Result<Self> {
        let lp = PreferenceLp::new(dim)?;
        Ok(Self {
            dijkstra,
            to_edges,
            from_edges,
            lp,
            _nid: std::marker::PhantomData,
            _eid: std::marker::PhantomData,
        })
    }

    pub fn shortcuts(
        &mut self,
        e1: &Edge<EID, NID>,
        e2: &Edge<EID, NID>,
    ) -> Result<Option<Shortcut<EID, NID>>> {
        if e1.from == e2.to {
            return Ok(None);
        }
        self.lp.reset()?;

        let mut shortcut_cost = e1.cost.clone();
        add_edge_costs(&mut shortcut_cost, &e2.cost);
        let shortcut_cost = shortcut_cost;

        let mut alpha = equal_weights(shortcut_cost.len());

        let shortcut = Shortcut {
            from: e1.from,
            to: e2.to,
            cost: shortcut_cost.clone(),
            replaced_edges: (e1.id, e2.id),
        };

        let mut exact = false;
        loop {
            let path_cost = (self.dijkstra)(e1.from, e2.to, &alpha);

            if is_dominated(&path_cost, &shortcut_cost) {
                return Ok(None);
            }

            if same_array(&path_cost, &shortcut_cost) {
                return Ok(Some(shortcut));
            }

            let constraint = path_cost
                .iter()
                .zip(shortcut_cost.iter())
                .map(|(p, s)| p - s)
                .collect::<Vec<_>>();

            self.lp.add_constraint(&constraint)?;

            match self.lp.solve(exact)? {
                Some((pref, delta)) => {
                    if delta + ACCURACY <= 0.0 {
                        return Ok(None);
                    } else if same_array(&pref, &alpha) {
                        if exact {
                            return Ok(Some(shortcut));
                        }
                        exact = true;
                        continue;
                    }
                    alpha = pref;
                    exact = false;
                }
                None => return Ok(None),
            }
        }
    }

    pub fn contract(&mut self, node: NID) -> Result<Vec<Shortcut<EID, NID>>> {
        let mut shortcuts = Vec::new();

        for to_edge in (self.to_edges)(node) {
            for from_edge in (self.from_edges)(node) {
                assert_eq!(
                    to_edge.to, from_edge.from,
                    "received edges that do not connect via the same node, {:?} != {:?}",
                    to_edge.to, from_edge.from
                );
                if let Some(shortcut) = self.shortcuts(&to_edge, &from_edge)? {
                    shortcuts.push(shortcut);
                }
            }
        }

        Ok(shortcuts)
    }
}

fn is_dominated(path_cost: &[f64], shortcut_cost: &[f64]) -> bool {
    let mut some_different = false;
    let dominated = !path_cost.iter().zip(shortcut_cost).any(|(p, s)| {
        if !float_eq!(p, s) {
            some_different = true;
        }
        p > s
    });
    dominated && some_different
}

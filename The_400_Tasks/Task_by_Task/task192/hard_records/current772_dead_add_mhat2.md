# task192 dead route: `m_hat2` Add DCE

- live bar: 772, parent sha `9edac4a076b7d58cf5eff45cd5bcc98bb911e1c3e0ce5be0a7de9f9023b6d2b1`
- attempted route: delete `n_mhat2` / `m_hat2` as an apparently dead Add after `m_hat`
- failed gate/check: raw graph operand dump before build; final `Einsum` input 26 is `m_hat2`
- why dead: `m_hat2` is load-bearing, not an unused value_info artifact; deleting it or replacing it with `m_hat` changes the final color-channel factor
- remaining hypothesis: algebraically absorb the single `E2[0,0]` increment into the final Einsum without materializing a full-output intermediate

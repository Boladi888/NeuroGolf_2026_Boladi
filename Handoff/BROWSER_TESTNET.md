# Browser Testnet Setting

The server-side agent profile is pinned to `base-sepolia` by `scripts/Set-HandoffTestnet.ps1`.

The HandOff website also has a browser-local network toggle stored in localStorage. If the UI ever appears to be on a non-testnet mode, open DevTools on `https://handoff.socnet.lol` and run:

```javascript
localStorage.setItem('handoff_testnet', '1');
localStorage.setItem('handoff_network', 'base-sepolia');
location.reload();
```

This is browser-local state. It does not expose or change credentials.

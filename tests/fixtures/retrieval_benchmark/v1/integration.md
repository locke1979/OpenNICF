# Integration contract

The order service sends a correlation ID to the access gateway and retries a transient endpoint failure once. The integration edge records both systems and the source version that described the route.

Diagnostics return a bounded evidence package with a locator, artifact hash, and correlation ID for follow-up.

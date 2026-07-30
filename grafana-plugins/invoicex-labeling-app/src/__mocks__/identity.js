// Stub for jest — returns empty proxy for any named import
const handler = { get: () => handler };
const proxy = new Proxy({}, handler);
module.exports = proxy;
module.exports.default = proxy;

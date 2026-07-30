/** @type {import('jest').Config} */
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  testMatch: ['**/__tests__/**/*.test.ts?(x)', '**/__tests__/**/*.test.js?(x)'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
    // Stub out Grafana + UI deps — not needed for pure-logic tests
    '^@grafana/(.*)$': '<rootDir>/src/__mocks__/identity',
    '^@emotion/(.*)$': '<rootDir>/src/__mocks__/identity',
    '^react-router-dom$': '<rootDir>/src/__mocks__/identity',
    '^react$': '<rootDir>/src/__mocks__/identity',
    '^react-dom$': '<rootDir>/src/__mocks__/identity',
  },
  // Polyfill window for api.ts which reads window.__INVOICEX_API_BASE__ at module load
  setupFiles: ['<rootDir>/src/__mocks__/setup.js'],
  transform: {
    '^.+\\.tsx?$': ['ts-jest', { tsconfig: 'tsconfig.json' }],
  },
};

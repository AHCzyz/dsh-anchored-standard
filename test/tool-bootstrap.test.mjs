import assert from 'node:assert/strict'
import test from 'node:test'

import { apply, name } from '../preset/tool-bootstrap.mjs'

const config = {
  commonTools: ['read'],
  shellTools: ['bash', 'pwsh'],
}

function register(cfg = config) {
  const listeners = new Map()
  const warns = []
  const ctx = {
    on(event, callback, options) {
      listeners.set(event, { callback, options })
    },
    logger: {
      warn(message) {
        warns.push(message)
      },
    },
  }
  apply(ctx, cfg)
  assert.equal(typeof listeners.get('system-prompt/assemble')?.callback, 'function')
  assert.equal(typeof listeners.get('agent/pre-step')?.callback, 'function')
  return { listeners, warns }
}

function assemble(listeners, events, tools, id = 's') {
  return listeners.get('system-prompt/assemble').callback(
    undefined,
    { agent: { session: { id, events } } },
    async () => ({ system: 'minimal persona', tools }),
  )
}

function preStep(listeners, events, messages, id = 's') {
  return listeners.get('agent/pre-step').callback(
    { agent: { session: { id, events } } },
    async () => ({ kind: 'enter', messages }),
  )
}

const userMessage = { id: 'u', content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' } }
const instructionMessage = { id: 'i', content: [], source: { kind: 'agent-instructions' } }
const catalogMessage = { id: 'c', content: [], source: { kind: 'skill-catalog' } }
const gestureMessage = { id: 'g', content: [], source: { kind: 'skill-invocation' } }
const pluginMessage = { id: 'p', content: [], source: { kind: 'plugin' } }

test('exports a diagnostic plugin name', () => {
  assert.equal(name, 'anchored-tool-bootstrap')
})

test('first request exposes one platform shell and read', async () => {
  const { listeners } = register()
  const tools = [{ name: 'pwsh' }, { name: 'read' }, { name: 'edit' }]
  const result = await assemble(listeners, [], tools)
  assert.deepEqual(result.tools.map((tool) => tool.name), ['pwsh', 'read'])
})

test('a durable tool call promotes the complete catalog', async () => {
  const { listeners } = register()
  const tools = [{ name: 'pwsh' }, { name: 'read' }, { name: 'edit' }, { name: 'grep' }]
  const result = await assemble(listeners, [{ type: 'tool/call', data: { name: 'read' } }], tools)
  assert.deepEqual(result.tools, tools)
})

test('a first assistant message promotes the complete catalog (no tool call needed)', async () => {
  const { listeners } = register()
  const tools = [{ name: 'pwsh' }, { name: 'read' }, { name: 'write' }]
  const result = await assemble(listeners, [{ type: 'assistant/message', data: {} }], tools)
  assert.deepEqual(result.tools, tools)
})

test('sessions derive promotion independently from their own events', async () => {
  const { listeners } = register()
  const tools = [{ name: 'bash' }, { name: 'read' }, { name: 'write' }]
  const promoted = await assemble(listeners, [{ type: 'tool/call' }], tools, 'a')
  const fresh = await assemble(listeners, [], tools, 'b')
  assert.deepEqual(promoted.tools, tools)
  assert.deepEqual(fresh.tools.map((tool) => tool.name), ['bash', 'read'])
})

test('promotion is memoized per session id within one process', async () => {
  const { listeners } = register()
  const tools = [{ name: 'bash' }, { name: 'read' }, { name: 'write' }]
  const first = await assemble(listeners, [{ type: 'tool/call' }], tools, 'memo')
  assert.deepEqual(first.tools, tools)
  // Same session id, events now empty: the cached decision still promotes.
  const second = await assemble(listeners, [], tools, 'memo')
  assert.deepEqual(second.tools, tools)
})

test('promoteOn tool-call requires a tool call, not just a reply', async () => {
  const { listeners } = register({ ...config, promoteOn: 'tool-call' })
  const tools = [{ name: 'bash' }, { name: 'read' }, { name: 'write' }]
  const replyOnly = await assemble(listeners, [{ type: 'assistant/message' }], tools, 'a')
  assert.deepEqual(replyOnly.tools.map((tool) => tool.name), ['bash', 'read'])
  const withCall = await assemble(listeners, [{ type: 'tool/call' }], tools, 'b')
  assert.deepEqual(withCall.tools, tools)
})

test('promoteOn assistant-message promotes after any first reply', async () => {
  const { listeners } = register({ ...config, promoteOn: 'assistant-message' })
  const tools = [{ name: 'bash' }, { name: 'read' }, { name: 'write' }]
  const result = await assemble(listeners, [{ type: 'assistant/message' }], tools, 'a')
  assert.deepEqual(result.tools, tools)
})

test('a missing bootstrap shell degrades gracefully to the full catalog', async () => {
  const { listeners, warns } = register()
  const tools = [{ name: 'read' }, { name: 'edit' }]
  const result = await assemble(listeners, [], tools)
  assert.deepEqual(result.tools, tools)
  assert.ok(warns.length >= 1)
})

test('invalid promoteOn values fail at apply time', () => {
  assert.throws(() => register({ ...config, promoteOn: 'bogus' }), /promoteOn/)
})

test('pre-step filter registers before every other listener', () => {
  const { listeners } = register()
  assert.equal(listeners.get('agent/pre-step').options?.prepend, true)
})

test('bootstrap strips auto-injected workspace and skill context', async () => {
  const { listeners } = register()
  const decision = await preStep(listeners, [], [
    userMessage,
    instructionMessage,
    catalogMessage,
    gestureMessage,
    pluginMessage,
  ])
  assert.equal(decision.kind, 'enter')
  assert.deepEqual(decision.messages.map((message) => message.id), ['u', 'g', 'p'])
})

test('a promoted session keeps every injected context message', async () => {
  const { listeners } = register()
  const messages = [userMessage, instructionMessage, catalogMessage, pluginMessage]
  const decision = await preStep(listeners, [{ type: 'tool/call' }], messages)
  assert.equal(decision.messages, messages)
})

test('a text-only first reply promotes the context injections too', async () => {
  const { listeners } = register()
  const stripped = await preStep(listeners, [], [userMessage, instructionMessage, catalogMessage])
  assert.deepEqual(stripped.messages.map((message) => message.id), ['u'])
  const kept = await preStep(listeners, [{ type: 'assistant/message' }], [userMessage, instructionMessage])
  assert.deepEqual(kept.messages.map((message) => message.id), ['u', 'i'])
})

test('rejected decisions pass through the context filter untouched', async () => {
  const { listeners } = register()
  const decision = { kind: 'reject', messages: [userMessage, instructionMessage] }
  const result = await listeners.get('agent/pre-step').callback(
    { agent: { session: { id: 's', events: [] } } },
    async () => decision,
  )
  assert.equal(result, decision)
})

test('suppressedContextSources is configurable', async () => {
  const { listeners } = register({ ...config, suppressedContextSources: ['skill-invocation'] })
  const decision = await preStep(listeners, [], [userMessage, instructionMessage, catalogMessage, gestureMessage])
  assert.deepEqual(decision.messages.map((message) => message.id), ['u', 'i', 'c'])
})

test('an empty suppressedContextSources disables the context filter', async () => {
  const { listeners } = register({ ...config, suppressedContextSources: [] })
  const messages = [userMessage, instructionMessage, catalogMessage]
  const decision = await preStep(listeners, [], messages)
  assert.equal(decision.messages, messages)
})

test('invalid suppressedContextSources values fail at apply time', () => {
  assert.throws(() => register({ ...config, suppressedContextSources: 'agent-instructions' }), /suppressedContextSources/)
  assert.throws(() => register({ ...config, suppressedContextSources: ['agent-instructions', 42] }), /suppressedContextSources/)
})

import assert from 'node:assert/strict'
import test from 'node:test'

import { apply, name } from '../preset/tool-bootstrap.mjs'

const config = {
  commonTools: ['read'],
  shellTools: ['bash', 'pwsh'],
}

function register(cfg = config) {
  const listeners = {}
  const warns = []
  const ctx = {
    on(event, callback) {
      listeners[event] = callback
    },
    logger: {
      warn(message) {
        warns.push(message)
      },
    },
  }
  apply(ctx, cfg)
  return { listeners, warns }
}

const agent = (events, id = 's') => ({ session: { id, events } })

function assemble(listener, events, tools, id = 's') {
  return listener(undefined, { agent: agent(events, id) }, async () => ({ system: 'minimal persona', tools }))
}

function request(listener, events, resolved, id = 's') {
  return listener({ agent: agent(events, id), turn: 1, step: 1 }, async () => resolved)
}

function prestep(listener, events, messages, id = 's') {
  return listener({ agent: agent(events, id), turn: 1, step: 1 }, async () => ({ kind: 'enter', messages }))
}

test('exports a diagnostic plugin name', () => {
  assert.equal(name, 'anchored-tool-bootstrap')
})

test('first request exposes one platform shell and read', async () => {
  const { listeners } = register()
  const tools = [{ name: 'pwsh' }, { name: 'read' }, { name: 'edit' }]
  const result = await assemble(listeners['system-prompt/assemble'], [], tools)
  assert.deepEqual(result.tools.map((tool) => tool.name), ['pwsh', 'read'])
})

test('a durable tool call promotes the complete catalog', async () => {
  const { listeners } = register()
  const tools = [{ name: 'pwsh' }, { name: 'read' }, { name: 'edit' }, { name: 'grep' }]
  const result = await assemble(listeners['system-prompt/assemble'], [{ type: 'tool/call', data: { name: 'read' } }], tools)
  assert.deepEqual(result.tools, tools)
})

test('a first assistant message promotes the complete catalog (no tool call needed)', async () => {
  const { listeners } = register()
  const tools = [{ name: 'pwsh' }, { name: 'read' }, { name: 'write' }]
  const result = await assemble(listeners['system-prompt/assemble'], [{ type: 'assistant/message', data: {} }], tools)
  assert.deepEqual(result.tools, tools)
})

test('sessions derive promotion independently from their own events', async () => {
  const { listeners } = register()
  const tools = [{ name: 'bash' }, { name: 'read' }, { name: 'write' }]
  const promoted = await assemble(listeners['system-prompt/assemble'], [{ type: 'tool/call' }], tools, 'a')
  const fresh = await assemble(listeners['system-prompt/assemble'], [], tools, 'b')
  assert.deepEqual(promoted.tools, tools)
  assert.deepEqual(fresh.tools.map((tool) => tool.name), ['bash', 'read'])
})

test('promotion is memoized per session id within one process', async () => {
  const { listeners } = register()
  const tools = [{ name: 'bash' }, { name: 'read' }, { name: 'write' }]
  const first = await assemble(listeners['system-prompt/assemble'], [{ type: 'tool/call' }], tools, 'memo')
  assert.deepEqual(first.tools, tools)
  // Same session id, events now empty: the cached decision still promotes.
  const second = await assemble(listeners['system-prompt/assemble'], [], tools, 'memo')
  assert.deepEqual(second.tools, tools)
})

test('promoteOn tool-call requires a tool call, not just a reply', async () => {
  const { listeners } = register({ ...config, promoteOn: 'tool-call' })
  const tools = [{ name: 'bash' }, { name: 'read' }, { name: 'write' }]
  const replyOnly = await assemble(listeners['system-prompt/assemble'], [{ type: 'assistant/message' }], tools, 'a')
  assert.deepEqual(replyOnly.tools.map((tool) => tool.name), ['bash', 'read'])
  const withCall = await assemble(listeners['system-prompt/assemble'], [{ type: 'tool/call' }], tools, 'b')
  assert.deepEqual(withCall.tools, tools)
})

test('promoteOn assistant-message promotes after any first reply', async () => {
  const { listeners } = register({ ...config, promoteOn: 'assistant-message' })
  const tools = [{ name: 'bash' }, { name: 'read' }, { name: 'write' }]
  const result = await assemble(listeners['system-prompt/assemble'], [{ type: 'assistant/message' }], tools, 'a')
  assert.deepEqual(result.tools, tools)
})

test('a missing bootstrap shell degrades gracefully to the full catalog', async () => {
  const { listeners, warns } = register()
  const tools = [{ name: 'read' }, { name: 'edit' }]
  const result = await assemble(listeners['system-prompt/assemble'], [], tools)
  assert.deepEqual(result.tools, tools)
  assert.ok(warns.length >= 1)
})

test('invalid promoteOn values fail at apply time', () => {
  assert.throws(() => register({ ...config, promoteOn: 'bogus' }), /promoteOn/)
})

test('invalid bootstrapMaxTokens fails at apply time', () => {
  assert.throws(() => register({ ...config, bootstrapMaxTokens: 0 }), /bootstrapMaxTokens/)
})

test('first request is capped to bootstrapMaxTokens', async () => {
  const { listeners } = register({ ...config, bootstrapMaxTokens: 1024 })
  const resolved = await request(listeners['agent/request'], [], { provider: 'deepseek-official', model: 'deepseek-v4-pro' })
  assert.equal(resolved.maxTokens, 1024)
  assert.equal(resolved.provider, 'deepseek-official')
})

test('after promotion, the injected cap is stripped so the default returns', async () => {
  const { listeners } = register({ ...config, bootstrapMaxTokens: 1024 })
  const resolved = await request(listeners['agent/request'], [{ type: 'tool/call' }], { provider: 'x', model: 'y', maxTokens: 1024 })
  assert.equal(resolved.maxTokens, undefined)
})

test('after promotion, a different maxTokens is preserved', async () => {
  const { listeners } = register({ ...config, bootstrapMaxTokens: 1024 })
  const resolved = await request(listeners['agent/request'], [{ type: 'tool/call' }], { provider: 'x', model: 'y', maxTokens: 256000 })
  assert.equal(resolved.maxTokens, 256000)
})

test('bootstrap pre-step strips skill-catalog and agent-instructions messages', async () => {
  const { listeners } = register()
  const messages = [
    { id: 'm1', content: [{ type: 'text', text: 'user message' }] },
    { id: 'm2', content: [{ type: 'text', text: '<system-reminder>skills...</system-reminder>' }], source: { kind: 'skill-catalog' } },
    { id: 'm3', content: [{ type: 'text', text: '# AGENTS.md content' }], source: { kind: 'agent-instructions' } },
  ]
  const decision = await prestep(listeners['agent/pre-step'], [], messages)
  assert.equal(decision.kind, 'enter')
  assert.deepEqual(decision.messages.map((message) => message.id), ['m1'])
})

test('promoted pre-step keeps skill-catalog and agent-instructions messages', async () => {
  const { listeners } = register()
  const messages = [
    { id: 'm1', content: [{ type: 'text', text: 'user message' }] },
    { id: 'm2', content: [{ type: 'text', text: '<system-reminder>skills...</system-reminder>' }], source: { kind: 'skill-catalog' } },
  ]
  const decision = await prestep(listeners['agent/pre-step'], [{ type: 'tool/call' }], messages)
  assert.deepEqual(decision.messages.map((message) => message.id), ['m1', 'm2'])
})

test('reject decisions pass through untouched', async () => {
  const { listeners } = register()
  const decision = await listeners['agent/pre-step']({ agent: agent([]), turn: 1, step: 1 }, async () => ({ kind: 'reject' }))
  assert.equal(decision.kind, 'reject')
})

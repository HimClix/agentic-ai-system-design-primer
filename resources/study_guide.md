# Agentic AI System Design — Study Guide

> How to use this resource based on your timeline and goals.

## 1-Week Sprint (Interview Prep)

**Goal:** Be able to design an agentic system on a whiteboard and discuss tradeoffs confidently.

| Day | Topic | Section |
|---|---|---|
| 1 | Agent basics: workflow vs agent, execution loop, single vs multi-agent | [Start Here](../README.md#agentic-ai-topics-start-here) |
| 2 | Agent patterns: ReAct, Plan-and-Execute, pattern selection guide | [Agent Patterns](../README.md#agent-architecture-patterns) |
| 3 | Memory systems + context engineering | [Memory](../README.md#memory-systems) + [Context](../README.md#context-engineering) |
| 4 | Tool architecture + RAG pipeline | [Tools](../README.md#tool-architecture) + [RAG](../README.md#rag-retrieval-augmented-generation) |
| 5 | Cost engineering + failure modes | [Cost](../README.md#cost-engineering) + [Failure Modes](../README.md#failure-modes--mitigation) |
| 6 | Practice: Customer Support Agent case study | [Solution](../solutions/customer_support_agent/) |
| 7 | Practice: AI Coding Agent case study + review interview questions | [Solution](../solutions/ai_coding_agent/) + [Appendix](../README.md#additional-interview-questions) |

## 2-Week Foundation

**Goal:** Solid understanding of all topics + able to discuss production concerns.

| Week | Focus | Sections |
|---|---|---|
| Week 1 | Core architecture (patterns, memory, tools, RAG, context) | All core topic sections |
| Week 2 | Production concerns (safety, observability, eval, cost, HITL, failure modes) + 4 case studies | All production sections + 4 solutions |

## 1-Month Comprehensive

**Goal:** Deep understanding of every topic, all case studies practiced, framework comparisons internalized.

| Week | Focus |
|---|---|
| Week 1 | Foundations: Start Here + Agent Patterns + Multi-Agent + Orchestration Frameworks |
| Week 2 | Building blocks: Memory + Context Engineering + RAG + Tool Architecture |
| Week 3 | Production: Safety + Observability + Eval + Cost + HITL + Failure Modes + Scale + Security |
| Week 4 | Applied: All 10 case studies + HLD/LLD + Interview questions + Papers |

## How to Practice Case Studies

For each case study in `/solutions/`:

1. **Read only the problem statement** (Step 1)
2. **Set a 45-minute timer** and design the system yourself on paper/whiteboard
3. **Then read the full solution** and compare your approach
4. Focus on: What tradeoffs did you miss? What patterns did you not consider?
5. **Present your design out loud** as if in an interview (builds verbal fluency)

## Key Papers (Read in Order)

1. [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — start here, always
2. [ReAct paper](https://huggingface.co/papers/2210.03629) — the foundational pattern
3. [MAST: Why Multi-Agent Systems Fail](https://arxiv.org/abs/2503.13657) — failure taxonomy
4. [Plan-then-Execute](https://arxiv.org/pdf/2509.08646) — the cost-efficient alternative to ReAct
5. [Mem0 paper](https://arxiv.org/pdf/2504.19413) — memory architecture benchmarks

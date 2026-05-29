# AI Product Manager Project Direction

This note records the long-term direction for optimizing this project for AI product manager interviews.

## Candidate Positioning

- Target role: AI Product Manager.
- The project should demonstrate ability to use AI tools fluently, understand AI capability boundaries, think in product terms, design useful user workflows, and create convincing AI-native product experiences.
- Future job descriptions may be added and should be used to tune the project narrative and feature priorities.

## Interview Feedback To Address

- The current frontend looks rough, lacks design quality, has poor UX, and feels functionally incomplete.
- The intelligent Q&A module produces weak model answers. Possible causes include poor data quality, weak retrieval context, insufficient prompt/product design, or unsuitable model usage.
- Pure natural-language Q&A is no longer differentiated enough. The project should evolve into an agent-style product that actively completes tasks and delivers business services.
- Current sentiment analysis and topic mining results have low real-world value. The project needs more reliable analysis methods, evaluation design, and business-facing insight outputs.

## New Product Positioning

Shift from a generic "birth-topic public opinion RAG Q&A demo" to an enterprise-facing AI agent platform for public opinion, market, and competitor intelligence.

Primary users:

- Brand operations teams: monitor brand mentions, collect public feedback across media platforms, identify risk signals, and generate response suggestions.
- Marketing and advertising teams: analyze campaign feedback, compare message performance, discover audience objections, and produce daily/weekly campaign insight briefs.
- Market and sales teams: track competitor movements, collect market signals, summarize customer pain points, and identify sales enablement opportunities.
- PR and customer experience teams: detect crises early, triage negative cases, trace issue sources, and prepare response playbooks.
- Management users: receive high-level dashboards, risk alerts, competitive summaries, and decision-ready reports.

## Product Design Principles

- Do not let current backend data limit frontend design. Use reserved modules, realistic mock data, and product prototypes when needed.
- The product should show AI product thinking, not only algorithm implementation.
- Prioritize workflows that solve real business jobs: monitoring, alerting, investigation, attribution, report generation, and action recommendation.
- Make the AI system explainable: show evidence, confidence, source comments, platform distribution, trend changes, and why an alert was triggered.
- Make capability boundaries visible: distinguish facts, inferred insights, model-generated recommendations, and low-confidence conclusions.
- Design for role-based experiences, where different teams see different tasks, metrics, and agent actions.

## Agent Product Direction

Potential agent capabilities:

- Monitoring Agent: continuously tracks brand keywords, competitor keywords, campaign names, product names, and crisis terms.
- Insight Agent: clusters public comments into business themes, summarizes drivers behind sentiment changes, and finds emerging issues.
- Competitor Agent: tracks competitor launches, price changes, campaign movements, user complaints, and share-of-voice shifts.
- Campaign Review Agent: evaluates campaign feedback, extracts user objections, compares platform performance, and suggests next creative angles.
- Risk Response Agent: detects risk spikes, gathers evidence, drafts response strategies, and recommends escalation levels.
- Report Agent: generates daily, weekly, or event-based reports with evidence, charts, and next actions.

## Analysis Improvement Direction

- Move beyond simple lexicon sentiment and basic LDA topic modeling.
- Consider stronger methods such as LLM-assisted labeling, fine-grained emotion taxonomy, aspect-based sentiment analysis, BERTopic, embedding clustering, supervised classifiers, and human-in-the-loop correction.
- Build evaluation around real product value: precision of risk alerts, usefulness of generated insights, evidence traceability, response time saved, and user satisfaction.
- Improve RAG by better chunking, metadata filters, source ranking, deduplication, recency weighting, prompt templates by task, and answer quality evaluation.

## Frontend Direction

- Build a polished enterprise SaaS-style interface, not a rough demo.
- Prefer role dashboards, task queues, alert centers, investigation workspaces, report builders, and agent activity timelines.
- Use realistic placeholder data and reserved modules to communicate product vision.
- The first screen should immediately show the product's business value and operating workflow.

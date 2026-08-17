# SirKit Task Draft: HandOff Workflow Audit for a Representative Technical Project

Created: 2026-07-03

## Intent

I want SirKit's HandOff expertise, not NeuroGolf competition expertise.

The attached packet is a compact example of a real technical project trying to use HandOff efficiently. Please treat NeuroGolf as representative workload context only: a project with many independent tasks, worker/reviewer roles, local automation, scarce external submit actions, and a user who needs to know what happened without reading every transcript.

Please do not spend context learning individual NeuroGolf puzzle/task details.

## What I Want From SirKit

Recommend the best way to use HandOff as it currently stands for this kind of project.

Focus on practical operating design:

1. How should I structure HandOff projects, goals, and tasks?
2. What should be its own project versus a goal or task inside an existing project?
3. What should go in task descriptions versus attached files versus artifacts/deliverables?
4. How should I make agent activity visible enough that I can tell whether someone did real work?
5. How should worker, reviewer, submit/release-captain, and admin/orchestrator responsibilities be split?
6. What minimum context packet should I attach so agents have enough context without wasting context window?
7. What HandOff platform friction or missing feature would most improve this workflow?

## Specific Representative Question

The packet includes a `Team_Submission/submit.py` flow from a sibling project. It is a scarce-resource release/submission gate:

- many agents can produce candidate outputs;
- a captain decides when to batch and submit;
- submits are limited and need safety checks;
- the action should be auditable and reversible where possible.

Should that kind of submit/release-captain flow be:

- its own HandOff project,
- a goal under the main project,
- a specialized task/subtask lane inside the main project,
- or modeled some other way?

Please generalize the answer beyond NeuroGolf. I want the pattern for any project that has independent workers plus a scarce external deployment/submission step.

## Requested Output

Please produce a concise but implementation-ready report with:

- recommended HandOff structure for this project class;
- suggested project/goal/task naming and lifecycle;
- what to put in deliverables versus artifacts versus proposals;
- a minimal context-packet template for future agent assignments;
- specific advice for the submit/release-captain pattern;
- how the requester should know an assigned agent actually did useful work;
- HandOff product feedback: top frictions, missing affordances, and likely quick wins.

If you create a durable artifact in HandOff, put the final report in the task's normal result/deliverable area. If the platform has a separate artifacts area, use artifacts for supporting notes and deliverables for the polished recommendation. Do not assume GitHub access.

## Boundaries

- Do not optimize NeuroGolf puzzle solving.
- Do not inspect or require secret credentials.
- Do not assume the recipient can access my local filesystem.
- Do not require GitHub access.
- Treat included code/docs as representative context for workflow design.

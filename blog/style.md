A LessWrong-style technical blog post is usually **argument-driven, explicit about uncertainty, and optimized for improving the reader’s model**, not just transmitting facts.

A good structure is:

## 1. Start with the actual problem

Do not begin with a vague intro. Begin with the thing that is confusing, counterintuitive, or under-modeled.

Weak:

> Today I want to talk about gradient descent.

Stronger:

> Gradient descent is often described as “following the slope downhill,” but that metaphor hides the most important fact: in high-dimensional spaces, almost every local move is a bet about geometry you cannot directly visualize.

The opening should make the reader think: “Yes, that is the thing I have been hand-waving.”

## 2. State your thesis early

LessWrong-style writing tends to reward clarity over suspense. Tell the reader what model you are going to argue for.

Example:

> My claim is that most debugging advice fails because it treats bugs as isolated mistakes, when in practice they are evidence about a flawed mental model of the system.

A technical post should usually have a claim like:

> X is better understood as Y.
> The standard explanation of X misses Z.
> Most failures of X come from hidden assumption Y.
> Here is a useful abstraction for thinking about X.

## 3. Build the idea from first principles

Instead of saying “obviously” or relying on authority, reconstruct the idea.

A common pattern:

1. Define the object.
2. Explain the naive model.
3. Show where the naive model breaks.
4. Introduce the better model.
5. Apply the better model to examples.

For example:

> The naive model of caching is: “store expensive results so we do not recompute them.”
> This is true, but incomplete. A cache is not just a performance optimization; it is also a second source of truth. Once you see it that way, cache invalidation stops looking like an implementation detail and starts looking like a consistency problem.

That move—**reframing the category**—is very LessWrong-ish.

## 4. Be explicit about definitions

LessWrong readers are unusually sensitive to overloaded words. Define terms before building arguments on them.

Instead of:

> The model generalizes poorly because it learned spurious features.

Write:

> By “spurious feature,” I mean a feature that is predictive in the training distribution but not stable under the distribution shift we care about.

Then use the definition consistently.

## 5. Use concrete examples early

Abstract arguments become much easier to follow when paired with toy cases.

A useful rhythm:

> Here is the general claim.
> Here is a toy example.
> Here is why the toy example captures the core structure.
> Here is where the toy example breaks down.

Example:

> Suppose you train a classifier to distinguish wolves from dogs, and all the wolf photos happen to contain snow. The classifier may learn “snow implies wolf.” This is not merely a funny dataset bug; it illustrates a deeper point about optimization pressure. Systems learn what is useful for the objective, not what we intended the objective to mean.

## 6. Separate claims by confidence level

LessWrong style often distinguishes between observations, hypotheses, intuitions, and stronger claims.

Use phrases like:

> I am confident that…
> My weaker claim is…
> My stronger suspicion is…
> I do not know whether this generalizes, but…
> This seems true in cases where…

This makes the writing feel more intellectually honest and less like marketing.

## 7. Show the mechanism

A technical LessWrong-style post should not merely say that something happens. It should explain **why the behavior emerges**.

Weak:

> LLMs hallucinate because they do not understand truth.

Stronger:

> One reason LLMs hallucinate is that next-token prediction rewards locally plausible continuations, while truthfulness is a global property of the completed answer. Unless the training process strongly penalizes false but fluent continuations, fluency and accuracy can come apart.

The key question is always:

> What causal process produces this outcome?

## 8. Use “gears-level” explanations

A gears-level explanation explains how the parts interact, not just what label applies.

Bad:

> The system failed because of Goodhart’s Law.

Better:

> The metric rewarded response speed. Teams optimized for response speed by closing tickets quickly. Once ticket closure became the target, it stopped tracking actual user resolution. That is the Goodhart failure: the proxy was optimized until it detached from the thing it originally measured.

Labels should come after the mechanism, not before it.

## 9. Include objections

A strong post argues with itself.

Add sections like:

> Objection: Isn’t this just X?
> Where this model fails
> A case where I would not use this abstraction
> The strongest counterargument I know

This is one of the fastest ways to make a technical post feel serious.

Example:

> Objection: you could say this is just overfitting.
> I think that is partly right, but “overfitting” is too broad here. The specific failure mode is not just fitting noise; it is fitting a feature that is genuinely predictive in one environment and actively misleading in another.

## 10. Prefer precise plain language over academic tone

LessWrong style is usually not formal academic prose. It is precise, but conversational.

Avoid:

> It is therefore evident that the aforementioned phenomenon constitutes a nontrivial epistemic hazard.

Prefer:

> This is dangerous because it gives you confidence exactly when your evidence has stopped tracking reality.

## 11. Make your reasoning inspectable

Do not hide leaps. Show intermediate steps.

Useful phrases:

> The reason I think this is…
> This implies…
> Notice that…
> The important part is not X, but Y.
> If this model is right, we should expect…

A LessWrong-style post often feels like watching someone debug their own thinking in public.

## 12. End with takeaways, not vibes

End by telling the reader what changed.

Good endings include:

> The main update is…
> The practical implication is…
> The model I want you to keep is…
> Questions I still do not know how to answer…

Example:

> The model I want you to keep is this: a cache is not just a speed trick. It is a partially synchronized replica of reality. Once you treat it that way, many “weird cache bugs” become ordinary distributed-systems problems.

## A reusable template

```markdown
# Title: [Concrete claim, not vague topic]

## The problem

[Describe the confusing thing or common failure mode.]

## The naive model

[Explain the standard explanation fairly.]

## Why the naive model is incomplete

[Show a counterexample, hidden assumption, or failure case.]

## A better model

[Introduce your abstraction or mechanism.]

## Example

[Walk through a simple concrete case.]

## What this predicts

[Say what should be true if your model is right.]

## Objections / limitations

[Steelman the counterargument.]

## Practical implications

[Explain how the reader should think or act differently.]

## Open questions

[Admit uncertainty and point to unresolved issues.]
```

## Mini example opening

```markdown
# Caches Are Distributed Systems in Disguise

Most programmers first learn caching as a performance trick: store the answer so you do not have to recompute it.

This is true, but it is not the important part.

The important part is that a cache creates a second place where reality is represented. Once there are two representations of reality, they can disagree. Once they can disagree, every cache bug becomes a consistency bug.

This post argues that cache invalidation is hard not because programmers are bad at remembering to clear things, but because caching quietly turns a local program into a small distributed system.
```

The core formula is:

> **Take a familiar technical concept, identify the hidden abstraction, explain the mechanism, test it against examples, and state where it breaks.**

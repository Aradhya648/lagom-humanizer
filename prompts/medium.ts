export function getMediumPrompt(inputText: string): string {
  return `Rewrite the text below so it reads like something a real person wrote. Keep all the ideas and information — don't cut anything or invent new points. But change how it's expressed.

Break up sentences that feel too polished or symmetrical. Vary the rhythm — not every paragraph needs to be the same length or follow the same pattern. Add a connective phrase when a transition feels abrupt. Drop one when it feels overexplained.

Avoid phrases that sound like they came from a template: "it is important to note", "in conclusion", "furthermore", "it is worth mentioning", "this demonstrates that". Replace them with whatever a person would actually say in that moment.

Don't write in a formal or clinical register unless the original specifically requires it. A little informality is fine — a contraction here, a shorter sentence there, a direct statement instead of a hedged one.

Output only the rewritten text. No intro, no explanation, nothing before or after.

---

${inputText}`;
}

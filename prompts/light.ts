export function getLightPrompt(inputText: string): string {
  return `Edit the text below lightly. The goal is to make it sound like a careful human writer worked through it — not a major rewrite, just enough to smooth out anything that feels overly stiff or formulaic.

Keep the structure, the argument order, and the overall tone. Don't add ideas or cut existing ones. Focus on the surface: vary a few sentence openings, replace phrases that sound like boilerplate, break up a sentence if it's running too long.

If something already sounds natural and clear, leave it alone. Only touch what needs it.

Academic or formal writing is fine — just make sure it sounds like a person with a command of the subject wrote it, not a template.

Output only the edited text. Nothing else.

---

${inputText}`;
}

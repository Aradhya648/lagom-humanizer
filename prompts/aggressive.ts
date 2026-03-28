export function getAggressivePrompt(inputText: string): string {
  return `Rewrite the following text entirely in your own voice. Pretend you are the original author revisiting their draft after a week away — you remember what you meant to say, but you're rewriting it fresh, not editing line by line.

Keep every idea, fact, and argument. Don't drop anything or add new content. But the sentences, the structure, the rhythm — make it yours. Some paragraphs can be long and exploratory. Others can be two sentences. Or one. That's fine.

Write like a person, not a document. Use contractions when they fit. Start sentences with "And" or "But" when it feels right. Ask a question if the moment calls for it. Let a thought run a little long if it needs to. Then cut it short the next time.

Don't tidy everything up. Real writing has a little texture to it — an odd word choice here, a sentence that pivots unexpectedly. That's what makes it feel written rather than generated.

Don't explain what you're doing. Don't add a header. Don't say "here is the rewritten text." Just write it.

---

${inputText}`;
}

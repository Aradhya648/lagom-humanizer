"use client";

import { useEffect, useState } from "react";

interface WordRevealProps {
  text: string;
}

export default function WordReveal({ text }: WordRevealProps) {
  const [displayedWords, setDisplayedWords] = useState<string[]>([]);
  const [key, setKey] = useState(0);

  useEffect(() => {
    if (!text) {
      setDisplayedWords([]);
      return;
    }

    setDisplayedWords([]);
    setKey((k) => k + 1);

    const words = text.split(" ");
    const batchSize = 8;
    let currentIndex = 0;

    const interval = setInterval(() => {
      if (currentIndex >= words.length) {
        clearInterval(interval);
        return;
      }
      const end = Math.min(currentIndex + batchSize, words.length);
      setDisplayedWords(words.slice(0, end));
      currentIndex = end;
    }, 30);

    return () => clearInterval(interval);
  }, [text]);

  if (!text) return null;

  const paragraphs = text.split(/\n\n+/);

  return (
    <div key={key} className="word-reveal text-[15px] leading-[1.8] text-text">
      {paragraphs.map((para, pIdx) => {
        const paraWords = para.split(" ");
        const prevWords = paragraphs
          .slice(0, pIdx)
          .reduce((acc, p) => acc + p.split(" ").length + 1, 0);

        return (
          <p key={pIdx} className={pIdx > 0 ? "mt-5" : ""}>
            {paraWords.map((word, wIdx) => {
              const globalIdx = prevWords + wIdx;
              const delay = Math.min(globalIdx * 14, 1800);
              return (
                <span
                  key={`${pIdx}-${wIdx}`}
                  style={{ animationDelay: `${delay}ms` }}
                >
                  {word}{wIdx < paraWords.length - 1 ? " " : ""}
                </span>
              );
            })}
          </p>
        );
      })}
    </div>
  );
}

import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Terms & Conditions — Lagom",
  description: "Terms and Conditions for Lagom by Drufiy AI Pvt. Ltd.",
};

export default function TermsPage() {
  return (
    <div className="min-h-screen flex flex-col bg-background text-text">

      {/* Header */}
      <header className="sticky top-0 z-20 bg-background/95 backdrop-blur-sm border-b border-border">
        <div className="max-w-3xl mx-auto px-5 sm:px-8 h-16 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2.5 hover:opacity-80 transition-opacity">
            <svg width="28" height="28" viewBox="0 0 32 32" fill="none" aria-hidden="true">
              <circle cx="16" cy="16" r="2.8" stroke="#22D3A0" strokeWidth="1.1" />
              <line x1="12.8" y1="16" x2="19.2" y2="16" stroke="#22D3A0" strokeWidth="0.7" opacity="0.45"/>
              <line x1="16" y1="12.8" x2="16" y2="19.2" stroke="#22D3A0" strokeWidth="0.7" opacity="0.45"/>
              {[0, 45, 90, 135, 180, 225, 270, 315].map((angle) => (
                <g key={angle} transform={`rotate(${angle} 16 16)`} opacity={angle % 90 === 0 ? 1 : 0.55}>
                  <line x1="15.3" y1="13.2" x2="16" y2="4.2" stroke="#22D3A0" strokeWidth="1.1" strokeLinecap="round"/>
                  <line x1="16.7" y1="13.2" x2="16" y2="4.2" stroke="#22D3A0" strokeWidth="1.1" strokeLinecap="round"/>
                  <circle cx="16" cy="4.2" r="0.9" fill="#22D3A0" />
                </g>
              ))}
            </svg>
            <span className="font-serif italic font-bold text-[1.4rem] text-text leading-none tracking-tight">lagom</span>
          </Link>
          <span className="text-xs text-faint">Terms &amp; Conditions</span>
        </div>
      </header>

      {/* Content */}
      <main className="flex-1 max-w-3xl mx-auto w-full px-5 sm:px-8 py-12">
        <div className="mb-10">
          <h1 className="text-3xl font-bold text-text mb-1">Terms &amp; Conditions</h1>
          <p className="text-xs text-faint">Last Updated: 22 April 2026</p>
        </div>

        <div className="space-y-8 text-[15px] leading-relaxed text-text/80">

          <section>
            <h2 className="text-base font-semibold text-text mb-2">1. Agreement to Terms</h2>
            <p>By accessing or using Lagom ("the Service") at golagom.up.railway.app, you agree to be bound by these Terms &amp; Conditions. If you do not agree, please discontinue use of the Service. These Terms apply to all visitors and users.</p>
          </section>

          <section>
            <h2 className="text-base font-semibold text-text mb-2">2. Who We Are</h2>
            <p>Lagom is operated by Drufiy AI Pvt. Ltd. ("Drufiy", "we", "us", or "our"), an incorporated AI products company registered in India. The Service is currently in beta (BetaV-0.1). Contact us at <a href="mailto:drufiyai001@gmail.com" className="text-accent hover:underline">drufiyai001@gmail.com</a> for any queries.</p>
          </section>

          <section>
            <h2 className="text-base font-semibold text-text mb-2">3. Description of Service</h2>
            <p className="mb-3">Lagom is an AI-powered text humanisation tool. You paste text into the Service, and it returns a rewritten version designed to read as natural human writing. Two modes are available:</p>
            <ul className="list-disc pl-5 space-y-1 text-text/70">
              <li><span className="text-text/90 font-medium">Fast Mode</span> — rapid single-pass rewriting.</li>
              <li><span className="text-text/90 font-medium">Deep Mode</span> — a more thorough multi-round process that scores your text against several external AI detection platforms before returning the final result.</li>
            </ul>
            <p className="mt-3">The Service is offered free of charge during the beta period. We reserve the right to introduce paid plans or usage limits at any time with reasonable notice.</p>
          </section>

          <section>
            <h2 className="text-base font-semibold text-text mb-2">4. Acceptable Use</h2>
            <p className="mb-3">You agree to use the Service only for lawful purposes. You must NOT use Lagom to:</p>
            <ul className="list-disc pl-5 space-y-1.5 text-text/70">
              <li>Submit, reproduce, or distribute content that infringes third-party intellectual property rights</li>
              <li>Generate or distribute spam, phishing content, or disinformation</li>
              <li>Circumvent AI detection for the purpose of academic fraud or plagiarism in violation of your institution's policies — use of the Service for academic submission is your sole responsibility</li>
              <li>Reverse-engineer, scrape, or exploit the Service or its infrastructure</li>
              <li>Submit content containing personal data of third parties without their consent</li>
              <li>Submit content that is defamatory, obscene, or otherwise unlawful</li>
            </ul>
            <p className="mt-3">We reserve the right to terminate or restrict access for any user who violates these Terms, without prior notice.</p>
          </section>

          <section>
            <h2 className="text-base font-semibold text-text mb-2">5. Intellectual Property</h2>
            <p className="mb-3">You retain full ownership of any text you submit to Lagom. By submitting text, you grant Drufiy a limited, non-exclusive, royalty-free licence to process that text solely for the purpose of delivering the Service to you.</p>
            <p>The Lagom platform — including its branding, design, and underlying technology — is the intellectual property of Drufiy AI Pvt. Ltd. You may not copy, reproduce, or distribute any part of the Service without prior written permission.</p>
          </section>

          <section>
            <h2 className="text-base font-semibold text-text mb-2">6. AI-Generated Output — Disclaimer</h2>
            <p className="mb-3">Lagom uses artificial intelligence to rewrite text. You acknowledge that:</p>
            <ul className="list-disc pl-5 space-y-1.5 text-text/70">
              <li>AI output may be imperfect, inaccurate, or unsuitable for your specific context</li>
              <li>We make no guarantees that rewritten text will bypass any particular AI detection system — detection tools update frequently and without notice</li>
              <li>You are solely responsible for reviewing, editing, and deciding whether to use any output produced by the Service</li>
            </ul>
          </section>

          <section>
            <h2 className="text-base font-semibold text-text mb-2">7. Third-Party Services</h2>
            <p className="mb-3">The Service relies on third-party providers to function. Their own terms and privacy policies govern their respective processing:</p>
            <ul className="list-disc pl-5 space-y-1.5 text-text/70">
              <li>Google (AI model provider)</li>
              <li>Render (hosting)</li>
              <li>Railway (backend infrastructure)</li>
              <li>Resend (error notifications)</li>
            </ul>
            <p className="mt-3">When you use Deep Mode, your text is submitted to a number of third-party AI detection platforms for scoring. By enabling Deep Mode, you explicitly consent to this submission. Do not submit confidential or sensitive information via Deep Mode.</p>
          </section>

          <section>
            <h2 className="text-base font-semibold text-text mb-2">8. Limitation of Liability</h2>
            <p className="mb-3">To the fullest extent permitted by applicable law, Drufiy AI Pvt. Ltd. shall not be liable for any indirect, incidental, consequential, or punitive damages arising from your use of — or inability to use — the Service. This includes loss of data, loss of profits, reputational harm, or any academic or professional consequences.</p>
            <p>Our total aggregate liability to you for any claim shall not exceed INR 0 (zero), reflecting that the Service is provided free of charge during the beta period.</p>
          </section>

          <section>
            <h2 className="text-base font-semibold text-text mb-2">9. Disclaimer of Warranties</h2>
            <p>The Service is provided "AS IS" and "AS AVAILABLE" without warranties of any kind, express or implied. We do not warrant that the Service will be uninterrupted, error-free, or free of harmful components. Beta software is inherently unstable — use it accordingly.</p>
          </section>

          <section>
            <h2 className="text-base font-semibold text-text mb-2">10. Modifications to Service and Terms</h2>
            <p>We reserve the right to modify, suspend, or discontinue the Service at any time without notice. We may update these Terms periodically. Continued use of the Service after changes constitutes acceptance of the revised Terms. The effective date at the top of this document reflects the most recent revision.</p>
          </section>

          <section>
            <h2 className="text-base font-semibold text-text mb-2">11. Governing Law</h2>
            <p>These Terms are governed by the laws of India. Any disputes shall be subject to the exclusive jurisdiction of the courts of New Delhi, India.</p>
          </section>

          <section>
            <h2 className="text-base font-semibold text-text mb-2">12. Contact</h2>
            <p>For questions or concerns regarding these Terms, contact us at: <a href="mailto:drufiyai001@gmail.com" className="text-accent hover:underline">drufiyai001@gmail.com</a></p>
          </section>

        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border py-5 mt-8">
        <div className="max-w-3xl mx-auto px-5 sm:px-8 flex items-center justify-between">
          <Link href="/" className="text-xs text-accent/60 hover:text-accent transition-colors">← Back to Lagom</Link>
          <span className="text-xs text-faint">© 2026 Drufiy AI Pvt. Ltd.</span>
        </div>
      </footer>
    </div>
  );
}

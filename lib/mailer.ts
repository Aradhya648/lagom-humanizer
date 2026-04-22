import { Resend } from "resend";

const NOTIFY = [
  "officialmaneeshawasthi@gmail.com",
  "officialaradhyamishra1@gmail.com",
];

export async function sendErrorEmail(
  technical: string,
  context: string,
  userFacing: string
): Promise<void> {
  const apiKey = process.env.RESEND_API_KEY;
  if (!apiKey) return;

  const resend = new Resend(apiKey);

  await resend.emails.send({
    from: "Lagom Errors <onboarding@resend.dev>",
    to: NOTIFY,
    subject: `[Lagom Error] ${context}`,
    html: `
      <div style="font-family:sans-serif;max-width:600px;margin:0 auto">
        <h2 style="color:#d32f2f">⚠ Lagom Humanizer — Error Alert</h2>
        <table style="width:100%;border-collapse:collapse">
          <tr><td style="padding:8px;font-weight:bold;width:140px">Context</td><td style="padding:8px">${context}</td></tr>
          <tr style="background:#f9f9f9"><td style="padding:8px;font-weight:bold">User saw</td><td style="padding:8px;color:#555">"${userFacing}"</td></tr>
          <tr><td style="padding:8px;font-weight:bold">Time (UTC)</td><td style="padding:8px">${new Date().toUTCString()}</td></tr>
        </table>
        <h3>Technical Error</h3>
        <pre style="background:#1e1e1e;color:#d4d4d4;padding:16px;border-radius:8px;overflow-x:auto;font-size:13px">${technical}</pre>
      </div>
    `,
  });
}

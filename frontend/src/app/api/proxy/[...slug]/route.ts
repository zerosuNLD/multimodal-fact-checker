import { NextRequest } from "next/server";

export async function GET(request: NextRequest, { params }: { params: Promise<{ slug: string[] }> }) {
  const resolvedParams = await params;
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const targetPath = resolvedParams.slug.join("/");
  const targetUrl = `${apiUrl}/${targetPath}`;

  const res = await fetch(targetUrl, {
    headers: {
      "ngrok-skip-browser-warning": "true",
      "Bypass-Tunnel-Reminder": "true",
    },
  });

  return new Response(res.body, {
    status: res.status,
    headers: {
      "Content-Type": res.headers.get("Content-Type") || "text/plain",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    },
  });
}

export async function POST(request: NextRequest, { params }: { params: Promise<{ slug: string[] }> }) {
  const resolvedParams = await params;
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const targetPath = resolvedParams.slug.join("/");
  const targetUrl = `${apiUrl}/${targetPath}`;

  const contentType = request.headers.get("content-type") || "";
  let body: any;

  if (contentType.includes("multipart/form-data")) {
    body = await request.formData();
  } else {
    body = await request.text();
  }

  const res = await fetch(targetUrl, {
    method: "POST",
    headers: {
      "ngrok-skip-browser-warning": "true",
      "Bypass-Tunnel-Reminder": "true",
      ...(contentType && !contentType.includes("multipart/form-data") ? { "Content-Type": contentType } : {}),
    },
    body,
  });

  return new Response(res.body, {
    status: res.status,
    headers: {
      "Content-Type": res.headers.get("Content-Type") || "application/json",
    },
  });
}

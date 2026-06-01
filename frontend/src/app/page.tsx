"use client";

import { useState, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Loader2, Upload, AlertCircle, CheckCircle2, X } from "lucide-react";
import { ErrorBoundary } from "./ErrorBoundary";

export default function Home() {
  const [claim, setClaim] = useState("");
  const [images, setImages] = useState<File[]>([]);
  const [language, setLanguage] = useState("en");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");

  type SSEEvent = { type: "step"; data: any } | { type: "token"; data: any };
  const [events, setEvents] = useState<SSEEvent[]>([]);
  const [isDone, setIsDone] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [isFeedbackOpen, setIsFeedbackOpen] = useState(false);
  const [feedback, setFeedback] = useState({ accuracy: "", reasoning: "", sources: "", image_understanding: "", comment: "" });
  const [isFeedbackSubmitted, setIsFeedbackSubmitted] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!claim.trim() && images.length === 0) {
      setError("Please provide a claim or an image to check.");
      return;
    }
    
    setIsSubmitting(true);
    setError("");
    setEvents([]);
    setIsDone(false);
    setIsFeedbackOpen(false);
    setIsFeedbackSubmitted(false);
    setFeedback({ accuracy: "", reasoning: "", sources: "", image_understanding: "", comment: "" });

    try {
      const formData = new FormData();
      formData.append("claim", claim);
      formData.append("language", language);
      images.forEach((img) => formData.append("images", img));

      const res = await fetch(`/api/proxy/api/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Failed to start analysis.");
      }

      const data = await res.json();
      setSessionId(data.session_id);
      const streamUrl = `/api/proxy${data.stream_url}`;

      const eventSource = new EventSource(streamUrl);

      eventSource.addEventListener("step", (e) => {
        setEvents((prev) => [...prev, { type: "step", data: JSON.parse(e.data) }]);
      });

      eventSource.addEventListener("token", (e) => {
        setEvents((prev) => [...prev, { type: "token", data: JSON.parse(e.data) }]);
      });

      eventSource.addEventListener("done", (e) => {
        setIsDone(true);
        eventSource.close();
        setIsSubmitting(false);
      });

      eventSource.addEventListener("error", (e) => {
        const parsed = JSON.parse((e as any).data || "{}");
        setError(parsed.message || "An error occurred during analysis.");
        setIsSubmitting(false);
        eventSource.close();
      });

      eventSource.addEventListener("heartbeat", (e) => {
        // Just keeping connection alive
      });

    } catch (err: any) {
      setError(err.message);
      setIsSubmitting(false);
    }
  };

  const submitFeedback = async () => {
    try {
      const res = await fetch(`/api/proxy/api/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          claim: claim || "N/A",
          ...feedback
        }),
      });
      if (res.ok) {
        setIsFeedbackSubmitted(true);
        setTimeout(() => setIsFeedbackOpen(false), 2000);
      }
    } catch (e) {
      console.error("Feedback error", e);
    }
  };

  const removeImage = (index: number) => {
    setImages((prev) => prev.filter((_, i) => i !== index));
  };

  // ── Parse Unified Timeline ──
  const timelineItems: { type: "step" | "thought"; title?: string; content: string; status?: string }[] = [];
  let currentLlmStream = "";
  let isAnswering = false;
  let finalAnswer = "";

  events.forEach((ev) => {
    if (ev.type === "step") {
      // Flush any accumulated thoughts from the current stream buffer
      const thoughtRegex = /✿THOUGHT✿[\s:]*(.*?)(?=(✿|$))/gs;
      let match;
      while ((match = thoughtRegex.exec(currentLlmStream)) !== null) {
        if (match[1].trim()) {
          timelineItems.push({ type: "thought", content: match[1].trim() });
        }
      }
      currentLlmStream = ""; // Clear buffer so we don't extract the same thoughts again

      if (ev.data.status === "completed") {
        for (let i = timelineItems.length - 1; i >= 0; i--) {
          if (timelineItems[i].type === "step" && timelineItems[i].title === ev.data.node) {
            timelineItems[i].status = "completed";
            break;
          }
        }
      } else {
        timelineItems.push({
          type: "step",
          title: ev.data.node,
          content: ev.data.output || ev.data.description || "Processing...",
          status: ev.data.status || "running"
        });
      }
    } else if (ev.type === "token") {
      if (isAnswering) {
        finalAnswer += ev.data.text;
      } else {
        currentLlmStream += ev.data.text;
        if (currentLlmStream.includes("✿RETURN✿")) {
          isAnswering = true;
          const parts = currentLlmStream.split("✿RETURN✿");
          const beforeReturn = parts[0];
          
          const thoughtRegex = /✿THOUGHT✿[\s:]*(.*?)(?=(✿|$))/gs;
          let match;
          while ((match = thoughtRegex.exec(beforeReturn)) !== null) {
            if (match[1].trim()) {
              timelineItems.push({ type: "thought", content: match[1].trim() });
            }
          }
          finalAnswer = parts.slice(1).join("✿RETURN✿").replace(/^:/, "").trimStart();
        }
      }
    }
  });

  // If not answered yet, extract ongoing thought
  if (!isAnswering && currentLlmStream) {
    const thoughtRegex = /✿THOUGHT✿[\s:]*(.*?)(?=(✿|$))/gs;
    let match;
    while ((match = thoughtRegex.exec(currentLlmStream)) !== null) {
      if (match[1].trim()) {
        timelineItems.push({ type: "thought", content: match[1].trim() });
      }
    }
  }

  const showThoughts = (isSubmitting || timelineItems.length > 0) && !isAnswering;

  return (
    <ErrorBoundary>
    <div className="min-h-screen bg-neutral-900 text-neutral-100 font-sans selection:bg-indigo-500/30 selection:text-indigo-200">
      <div className="max-w-6xl mx-auto px-4 py-12">
        <header className="mb-12 text-center">
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
            AI Fact Checker
          </h1>
          <p className="mt-4 text-neutral-400 max-w-2xl mx-auto text-lg">
            Verify claims and analyze images in real-time. Uncover the truth with our advanced reasoning pipeline.
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Input Section */}
          <div className="lg:col-span-5 space-y-6">
            <div className="bg-neutral-800/50 backdrop-blur-xl border border-neutral-700/50 p-6 rounded-2xl shadow-xl">
              <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
                <span className="bg-indigo-500/20 text-indigo-400 p-2 rounded-lg">
                  <CheckCircle2 size={20} />
                </span>
                Submit a Query
              </h2>
              <form onSubmit={handleSubmit} className="space-y-5">
                <div>
                  <label className="block text-sm font-medium text-neutral-300 mb-2">Claim to Verify</label>
                  <textarea
                    rows={4}
                    value={claim}
                    onChange={(e) => setClaim(e.target.value)}
                    className="w-full bg-neutral-900 border border-neutral-700 rounded-xl px-4 py-3 text-neutral-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all placeholder:text-neutral-600 resize-none"
                    placeholder="E.g., Does coffee cause dehydration?"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-neutral-300 mb-2">Language</label>
                  <select
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                    className="w-full bg-neutral-900 border border-neutral-700 rounded-xl px-4 py-3 text-neutral-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all appearance-none"
                  >
                    <option value="en">English</option>
                    <option value="vi">Vietnamese</option>
                    <option value="es">Spanish</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-neutral-300 mb-2">Upload Images</label>
                  <div 
                    onClick={() => fileInputRef.current?.click()}
                    className="border-2 border-dashed border-neutral-700 hover:border-indigo-500/50 bg-neutral-900/50 rounded-xl p-6 text-center cursor-pointer transition-all group"
                  >
                    <Upload className="mx-auto h-8 w-8 text-neutral-500 group-hover:text-indigo-400 transition-colors mb-3" />
                    <p className="text-sm text-neutral-400 group-hover:text-neutral-300">Click to browse or drag and drop</p>
                    <p className="text-xs text-neutral-500 mt-1">JPG, PNG, WEBP (Max 5MB)</p>
                    <input
                      type="file"
                      ref={fileInputRef}
                      className="hidden"
                      multiple
                      accept="image/*"
                      onChange={(e) => {
                        if (e.target.files) {
                          setImages((prev) => [...prev, ...Array.from(e.target.files!)]);
                        }
                      }}
                    />
                  </div>
                  
                  {images.length > 0 && (
                    <div className="mt-4 flex flex-wrap gap-3">
                      {images.map((file, i) => (
                        <div key={i} className="relative group rounded-lg overflow-hidden border border-neutral-700 bg-neutral-900 h-16 w-24 flex items-center justify-center">
                          <img src={URL.createObjectURL(file)} alt="preview" className="object-cover w-full h-full opacity-70 group-hover:opacity-100 transition-opacity" />
                          <button
                            type="button"
                            onClick={() => removeImage(i)}
                            className="absolute top-1 right-1 bg-black/60 p-1 rounded-full text-white opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-500"
                          >
                            <X size={14} />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {error && (
                  <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-xl flex items-start gap-3 text-sm">
                    <AlertCircle size={18} className="mt-0.5 shrink-0" />
                    <p>{error}</p>
                  </div>
                )}

                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-3.5 px-6 rounded-xl shadow-[0_0_20px_rgba(79,70,229,0.3)] hover:shadow-[0_0_25px_rgba(79,70,229,0.5)] transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 className="animate-spin" size={18} />
                      Analyzing...
                    </>
                  ) : (
                    "Fact Check Now"
                  )}
                </button>
              </form>
            </div>
          </div>

          {/* Results Section */}
          <div className="lg:col-span-7">
            <div className="bg-neutral-800/50 backdrop-blur-xl border border-neutral-700/50 rounded-2xl shadow-xl h-full min-h-[500px] flex flex-col overflow-hidden relative">
              <div className="p-6 border-b border-neutral-700/50 flex justify-between items-center bg-neutral-800/80">
                <h2 className="text-xl font-semibold flex items-center gap-2">
                  <span className="bg-cyan-500/20 text-cyan-400 p-2 rounded-lg">
                    <CheckCircle2 size={20} />
                  </span>
                  Analysis Results
                </h2>
                {isSubmitting && !isDone && (
                  <div className="flex items-center gap-2 text-sm text-cyan-400 bg-cyan-400/10 px-3 py-1.5 rounded-full border border-cyan-400/20 shadow-[0_0_10px_rgba(34,211,238,0.2)]">
                    <span className="relative flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
                    </span>
                    Agent Active
                  </div>
                )}
              </div>

              <div className="p-6 flex-1 overflow-y-auto">
                {/* Initial State */}
                {!isSubmitting && timelineItems.length === 0 && !finalAnswer && !error && (
                  <div className="h-full flex flex-col items-center justify-center text-neutral-500 gap-4 opacity-50">
                    <div className="w-16 h-16 rounded-2xl bg-neutral-800 border border-neutral-700 flex items-center justify-center">
                      <CheckCircle2 size={32} />
                    </div>
                    <p className="text-center max-w-sm">
                      Submit a query on the left to start the AI fact-checking process.
                    </p>
                  </div>
                )}

                {/* Show thoughts only when not answering yet */}
                {showThoughts && (
                  <div className="space-y-4 animate-in fade-in duration-500">
                    <div className="flex items-center gap-3 text-indigo-400 mb-6 font-medium bg-indigo-500/10 p-3 rounded-lg border border-indigo-500/20">
                      <Loader2 className="animate-spin" size={18} />
                      Agent is reasoning...
                    </div>
                    
                    <div className="relative pl-6 border-l-2 border-neutral-700/50 space-y-6 ml-2">
                      {timelineItems.map((item, idx) => (
                        <div key={`timeline-${idx}`} className="relative">
                          {item.type === "step" ? (
                            <>
                              <div className={`absolute -left-[31px] border-2 w-4 h-4 rounded-full z-10 flex items-center justify-center ${
                                item.status === "completed" 
                                  ? "bg-emerald-500 border-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]" 
                                  : "bg-neutral-900 border-neutral-500"
                              }`}>
                                {item.status === "completed" && <CheckCircle2 size={10} className="text-white" />}
                              </div>
                              <div className={`bg-neutral-800/50 border rounded-xl p-4 text-sm shadow-sm transition-colors ${
                                item.status === "completed" ? "border-emerald-500/30" : "border-neutral-700/30"
                              }`}>
                                <div className={`text-xs font-semibold mb-1 uppercase tracking-wider ${
                                  item.status === "completed" ? "text-emerald-400" : "text-neutral-400"
                                }`}>{item.title}</div>
                                <div className="text-neutral-300">{item.content}</div>
                              </div>
                            </>
                          ) : (
                            <>
                              <div className="absolute -left-[31px] bg-neutral-900 border-2 border-indigo-500 w-4 h-4 rounded-full shadow-[0_0_10px_rgba(79,70,229,0.5)] z-10" />
                              <div className="bg-indigo-500/5 border border-indigo-500/20 rounded-xl p-4 text-sm shadow-sm hover:border-indigo-500/40 transition-colors">
                                <div className="text-xs text-indigo-400 font-semibold mb-1 uppercase tracking-wider">THOUGHT</div>
                                <div className="text-neutral-200">{item.content}</div>
                              </div>
                            </>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Show final answer once it starts streaming */}
                {isAnswering && (
                  <div className="animate-in fade-in zoom-in-95 duration-500">
                    <div className="prose prose-invert prose-indigo max-w-none whitespace-pre-wrap">
                      <ReactMarkdown 
                        remarkPlugins={[remarkGfm]}
                        components={{
                          a: ({node, ...props}) => (
                            <a {...props} target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:text-cyan-300 underline underline-offset-2 break-all" />
                          )
                        }}
                      >
                        {finalAnswer.replace(/^[\s:]+/, "")}
                      </ReactMarkdown>
                    </div>
                    {isSubmitting && !isDone && (
                      <div className="mt-4 flex items-center gap-2 text-neutral-400 text-sm">
                        <Loader2 className="animate-spin" size={14} />
                        Generating final report...
                      </div>
                    )}
                  </div>
                )}
                {/* Feedback Panel */}
                {isDone && (
                  <div className="mt-8 border-t border-white/10 pt-6">
                    <button
                      onClick={() => setIsFeedbackOpen(!isFeedbackOpen)}
                      className="text-sm font-medium text-neutral-300 hover:text-white transition-colors flex items-center gap-2 bg-white/5 px-4 py-2 rounded-full border border-white/10"
                    >
                      {isFeedbackOpen ? "Thu gọn đánh giá" : "Đánh giá câu trả lời này"}
                      <span className="text-lg">{isFeedbackOpen ? "↑" : "↓"}</span>
                    </button>

                    {isFeedbackOpen && !isFeedbackSubmitted && (
                      <div className="mt-4 p-5 rounded-xl border border-white/10 bg-black/40 backdrop-blur-md animate-in fade-in slide-in-from-top-4 duration-300 space-y-5">
                        <h4 className="font-semibold text-white">Phản hồi của bạn giúp AI tốt hơn</h4>
                        
                        <div className="space-y-2">
                          <label className="text-sm text-neutral-400">1. Mức độ chính xác (Accuracy)</label>
                          <div className="flex flex-wrap gap-2">
                            {["Chính xác", "Đúng một phần", "Sai lệch", "Không thể xác minh"].map(opt => (
                              <button key={opt} onClick={() => setFeedback({...feedback, accuracy: opt})} className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${feedback.accuracy === opt ? "bg-indigo-500 text-white" : "bg-white/5 text-neutral-300 hover:bg-white/10"}`}>{opt}</button>
                            ))}
                          </div>
                        </div>

                        <div className="space-y-2">
                          <label className="text-sm text-neutral-400">2. Chất lượng lập luận (Reasoning)</label>
                          <div className="flex flex-wrap gap-2">
                            {["Rõ ràng, logic", "Chấp nhận được", "Lan man, khó hiểu"].map(opt => (
                              <button key={opt} onClick={() => setFeedback({...feedback, reasoning: opt})} className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${feedback.reasoning === opt ? "bg-indigo-500 text-white" : "bg-white/5 text-neutral-300 hover:bg-white/10"}`}>{opt}</button>
                            ))}
                          </div>
                        </div>

                        <div className="space-y-2">
                          <label className="text-sm text-neutral-400">3. Độ uy tín của nguồn (Sources)</label>
                          <div className="flex flex-wrap gap-2">
                            {["Rất uy tín", "Bình thường", "Không đáng tin cậy", "Không có nguồn"].map(opt => (
                              <button key={opt} onClick={() => setFeedback({...feedback, sources: opt})} className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${feedback.sources === opt ? "bg-indigo-500 text-white" : "bg-white/5 text-neutral-300 hover:bg-white/10"}`}>{opt}</button>
                            ))}
                          </div>
                        </div>

                        {images.length > 0 && (
                          <div className="space-y-2">
                            <label className="text-sm text-neutral-400">4. Khả năng hiểu hình ảnh</label>
                            <div className="flex flex-wrap gap-2">
                              {["Nhận diện tốt", "Bỏ sót chi tiết", "Nhận diện sai"].map(opt => (
                                <button key={opt} onClick={() => setFeedback({...feedback, image_understanding: opt})} className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${feedback.image_understanding === opt ? "bg-indigo-500 text-white" : "bg-white/5 text-neutral-300 hover:bg-white/10"}`}>{opt}</button>
                              ))}
                            </div>
                          </div>
                        )}

                        <div className="space-y-2">
                          <label className="text-sm text-neutral-400">Góp ý chi tiết (Tùy chọn)</label>
                          <textarea 
                            value={feedback.comment}
                            onChange={(e) => setFeedback({...feedback, comment: e.target.value})}
                            placeholder="Nhập thêm nhận xét của bạn..."
                            className="w-full h-20 bg-white/5 border border-white/10 rounded-lg p-3 text-sm text-white placeholder-neutral-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 resize-none"
                          />
                        </div>

                        <button 
                          onClick={submitFeedback}
                          className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-colors"
                        >
                          Gửi Phản Hồi
                        </button>
                      </div>
                    )}
                    
                    {isFeedbackSubmitted && (
                      <div className="mt-4 p-4 rounded-xl border border-green-500/20 bg-green-500/10 text-green-400 text-sm flex items-center justify-center gap-2 animate-in zoom-in-95 duration-300">
                        <span>✓</span> Cảm ơn bạn đã gửi đánh giá!
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    </ErrorBoundary>
  );
}

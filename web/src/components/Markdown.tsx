import { Check, Copy } from "lucide-react";
import { memo, useState } from "react";
import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";

function extractText(node: ReactNode): string {
  if (node === null || node === undefined || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(extractText).join("");
  const element = node as { props?: { children?: ReactNode } };
  return element.props ? extractText(element.props.children) : "";
}

function CopyButton({ getText }: { getText: () => string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      aria-label="Copy code"
      className="absolute right-2 top-2 rounded-md border bg-zinc-900 p-1 text-zinc-400 opacity-0 transition group-hover:opacity-100"
      onClick={() => {
        void navigator.clipboard.writeText(getText()).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        });
      }}
    >
      {copied ? <Check size={13} /> : <Copy size={13} />}
    </button>
  );
}

const Markdown = memo(function Markdown({ text }: { text: string }) {
  return (
    <div className="prose prose-sm max-w-none break-words dark:prose-invert">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          pre({ children, ...props }) {
            return (
              <div className="group relative my-3 overflow-hidden rounded-xl border bg-zinc-950">
                <CopyButton getText={() => extractText(children)} />
                <pre className="overflow-x-auto p-4 text-[13px]" {...props}>
                  {children}
                </pre>
              </div>
            );
          },
          code({ children, className }) {
            if (className?.startsWith("language-")) {
              return (
                <code className={`${className} font-mono`} style={{ whiteSpace: "pre" }}>
                  {children}
                </code>
              );
            }
            return (
              <code className="rounded bg-zinc-100 px-1 py-0.5 font-mono text-[13px] dark:bg-zinc-800">
                {String(children)}
              </code>
            );
          },
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
});

export default Markdown;

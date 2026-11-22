// Flattens OpenAI message content (a string or an array of parts) to plain text.
//
// Venice is reached through its text chat, so images, files and audio cannot be
// passed on. Dropping such a part silently leaves the model guessing: opencode
// sends an image it read as a tool result "Image read successfully" followed by
// a user message holding only the image_url part, and with the part gone the
// model went on to describe screenshots it never saw. Each such part becomes a
// short note in the text instead, so the model knows it is missing.
export function normalizeOpenAiContent(content) {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") {
          return part;
        }
        if ((part?.type === "text" || part?.type === "input_text") && typeof part.text === "string") {
          return part.text;
        }
        return omittedMediaNote(part);
      })
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

// Number of image, file and audio parts in a list of OpenAI messages, i.e. how
// many notes normalizeOpenAiContent puts in their place.
export function countOmittedMedia(messages) {
  let count = 0;
  for (const message of Array.isArray(messages) ? messages : []) {
    if (Array.isArray(message?.content)) {
      for (const part of message.content) {
        if (describeMediaPart(part)) {
          count += 1;
        }
      }
    }
  }
  return count;
}

const NOTE_DETAIL_MAX_CHARS = 200;
// A cut URL cannot be fetched, so a longer one is left out instead.
const NOTE_URL_MAX_CHARS = 2000;

// { kind, detail } for an image, file or audio part (Chat Completions and
// Responses shapes), or null for anything else. The detail names what was
// left out: a data URL's media type, a web URL the model could fetch with its
// own tools, or a file name.
function describeMediaPart(part) {
  switch (part?.type) {
    case "image_url":
    case "input_image": {
      const url = typeof part.image_url === "string" ? part.image_url : part.image_url?.url;
      return { kind: "image", detail: describeUrl(url) };
    }
    case "file":
    case "input_file": {
      const file = part.type === "file" ? part.file : part;
      const name = file?.filename;
      return { kind: "file", detail: typeof name === "string" ? cleanDetail(name) : describeUrl(file?.file_url) };
    }
    case "input_audio":
      return { kind: "audio", detail: cleanDetail(part.input_audio?.format ?? "") };
    default:
      return null;
  }
}

function describeUrl(url) {
  if (typeof url !== "string") {
    return "";
  }
  const dataUrl = /^data:([\w.+-]+\/[\w.+-]+)[;,]/i.exec(url);
  if (dataUrl) {
    return dataUrl[1].toLowerCase();
  }
  if (!/^https?:\/\/\S+$/i.test(url) || url.length > NOTE_URL_MAX_CHARS) {
    return "";
  }
  return url.replace(/[[\]]/g, encodeURIComponent);
}

// One line, no brackets that could end the note early, bounded length.
function cleanDetail(value) {
  if (typeof value !== "string") {
    return "";
  }
  const line = value.replace(/[\s\u0000-\u001f[\]]+/g, " ").trim();
  return line.length > NOTE_DETAIL_MAX_CHARS ? `${line.slice(0, NOTE_DETAIL_MAX_CHARS)}…` : line;
}

// The note that stands in for an image, file or audio part, or "" for any
// other part. Also used by the agent endpoint (agentic.mjs).
export function omittedMediaNote(part) {
  const media = describeMediaPart(part);
  if (!media) {
    return "";
  }
  const { kind, detail } = media;
  const label = kind[0].toUpperCase() + kind.slice(1);
  return `[${label} omitted${detail ? ` (${detail})` : ""}: this connection carries text only, so the ${kind} was not delivered and you cannot see its content; reading or sending it again will not change that. Do not guess what it contains.]`;
}

export interface User {
  id: number;
  username: string;
  display_name: string;
  language: string;
}

export interface Source {
  document_id: string;
  document_title: string;
  chapter: string;
  course_code: string;
  kind: string;
  language: string;
  entity_type?: string;
  entity_id?: string;
}

export interface Citation {
  marker: string;
  source: Source;
}

export interface Turn {
  role: string;
  text: string;
  citations: Citation[];
  refused: boolean;
  rephrase_suggestion: string;
  skills_applied: string[];
}

export interface ConversationMeta {
  id: string;
  user_id: number;
  title: string;
  created_at: string;
  updated_at: string;
  preview: string;
}

export interface Attachment {
  id: string;
  conversation_id: string;
  filename: string;
  file_kind: string;
  size_bytes: number;
  chunk_count: number;
  created_at: string;
}

export interface Features {
  has_web_search: boolean;
}

export interface EntityDetail {
  [key: string]:
    | string
    | number
    | boolean
    | EntityDetail[]
    | { [k: string]: string | number | boolean };
}

export interface Entity {
  id: string;
  type: string;
  name: string;
  detail: EntityDetail;
}

export interface EntityBundle {
  departments: Entity[];
  instructors: Entity[];
  programs: Entity[];
  terms: Entity[];
}
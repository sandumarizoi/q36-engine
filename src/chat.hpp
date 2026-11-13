// q36: the Qwen3.6 chat template (same output as the Jinja template in the GGUF, text only) and tool-call parsing.
#pragma once
#include "json.hpp"
#include <cstring>

namespace q36 {

static inline std::string trim(const std::string & s) {
    size_t a = 0, b = s.size(); while (a < b && strchr(" \t\r\n", s[a])) a++; while (b > a && strchr(" \t\r\n", s[b - 1])) b--; return s.substr(a, b - a);
}
static inline std::string content_text(const J & c) { // string, or array of {type:text,text}
    if (c.is_str()) return c.s;
    std::string o;
    if (c.is_arr()) for (auto & it : c.a) { if (it.find("text")) o += it.at("text").str(); else throw std::runtime_error("only text content is supported"); }
    return o;
}
static inline bool starts(const std::string & s, const char * p) { return s.compare(0, strlen(p), p) == 0; }
static inline bool ends(const std::string & s, const char * p) { const size_t n = strlen(p); return s.size() >= n && s.compare(s.size() - n, n, p) == 0; }

static inline std::string render_chat(const J & messages, const J & tools, bool enable_thinking, bool preserve_thinking, bool add_generation_prompt = true) {
    const auto & m = messages.a; if (m.empty()) throw std::runtime_error("No messages provided.");
    auto role = [&](size_t i) { return m[i].at("role").str(); };
    auto is_sys = [&](size_t i) { return role(i) == "system" || role(i) == "developer"; };
    size_t num_sys = 0; std::string merged, out;
    if (is_sys(0)) {
        merged = trim(content_text(m[0].at("content"))); num_sys = 1;
        if (m.size() > 1 && is_sys(1)) { merged += "\n" + trim(content_text(m[1].at("content"))); num_sys = 2; }
    }
    if (tools.is_arr() && !tools.a.empty()) {
        out += "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (auto & t : tools.a) { out += "\n"; out += t.dump(true); }
        out += "\n</tools>";
        out += "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";
        if (!merged.empty()) out += "\n\n" + merged;
        out += "<|im_end|>\n";
    } else if (!merged.empty()) out += "<|im_start|>system\n" + merged + "<|im_end|>\n";
    // the last real user query (a user message that is not a tool response)
    size_t last_query = m.size() - 1;
    for (size_t k = m.size(); k-- > 0;) {
        if (role(k) == "user") { const std::string c = trim(content_text(m[k].at("content"))); if (!(starts(c, "<tool_response>") && ends(c, "</tool_response>"))) { last_query = k; break; } }
    }
    for (size_t i = 0; i < m.size(); i++) {
        if (i < num_sys || is_sys(i)) continue;
        std::string content = trim(content_text(m[i].at("content"))); const std::string r = role(i);
        if (r == "user") out += "<|im_start|>user\n" + content + "<|im_end|>\n";
        else if (r == "assistant") {
            std::string reasoning;
            if (m[i].at("reasoning_content").is_str()) reasoning = m[i].at("reasoning_content").s;
            else { const size_t e = content.find("</think>");
                if (e != std::string::npos) {
                    std::string head = content.substr(0, e); while (!head.empty() && head.back() == '\n') head.pop_back();
                    const size_t b = head.rfind("<think>"); if (b != std::string::npos) head = head.substr(b + 7); size_t k = 0; while (k < head.size() && head[k] == '\n') k++; reasoning = head.substr(k);
                    const size_t le = content.rfind("</think>"); content = content.substr(le + 8); k = 0; while (k < content.size() && content[k] == '\n') k++; content = content.substr(k);
                } }
            reasoning = trim(reasoning);
            if (preserve_thinking || i > last_query) out += "<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content;
            else out += "<|im_start|>assistant\n" + content;
            const J & tcs = m[i].at("tool_calls");
            if (tcs.is_arr()) for (size_t k = 0; k < tcs.a.size(); k++) {
                const J & tc = tcs.a[k].find("function") ? tcs.a[k].at("function") : tcs.a[k];
                if (k == 0) out += trim(content).empty() ? "<tool_call>\n<function=" : "\n\n<tool_call>\n<function="; else out += "\n<tool_call>\n<function=";
                out += tc.at("name").str() + ">\n";
                J args = tc.at("arguments"); if (args.is_str()) { J p; if (J::try_parse(args.s, p)) args = p; }
                if (args.is_obj()) for (auto & kv : args.o) {
                    out += "<parameter=" + kv.first + ">\n";
                    if (kv.second.is_str()) out += kv.second.s; else if (kv.second.t == J::Bool) out += kv.second.b ? "True" : "False"; else if (kv.second.is_null()) out += "None"; else out += kv.second.dump(true);
                    out += "\n</parameter>\n";
                }
                out += "</function>\n</tool_call>";
            }
            out += "<|im_end|>\n";
        } else if (r == "tool") {
            if (i > 0 && role(i - 1) != "tool") out += "<|im_start|>user";
            out += "\n<tool_response>\n" + content + "\n</tool_response>";
            if (i + 1 == m.size() || role(i + 1) != "tool") out += "<|im_end|>\n";
        }
    }
    if (add_generation_prompt) out += enable_thinking ? "<|im_start|>assistant\n<think>\n" : "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    return out;
}

// "<function=name>\n<parameter=k>\nvalue\n</parameter>\n...</function>" -> {name, arguments (JSON text)}; types follow the tool schema
static inline bool parse_tool_call(const std::string & body, const J & tools, std::string & name, std::string & args_json) {
    const size_t f = body.find("<function="); if (f == std::string::npos) return false;
    const size_t fe = body.find('>', f); if (fe == std::string::npos) return false;
    name = trim(body.substr(f + 10, fe - f - 10));
    const J * props = nullptr;
    if (tools.is_arr()) for (auto & t : tools.a) { const J & fn = t.find("function") ? t.at("function") : t; if (fn.at("name").str() == name) { const J * pr = fn.at("parameters").find("properties"); if (pr) props = pr; } }
    J args = J::obj(); size_t pos = fe + 1;
    for (;;) {
        const size_t p = body.find("<parameter=", pos); if (p == std::string::npos) break;
        const size_t pe = body.find('>', p); if (pe == std::string::npos) break;
        const std::string key = trim(body.substr(p + 11, pe - p - 11));
        size_t ve = body.find("</parameter>", pe); if (ve == std::string::npos) ve = body.size();
        std::string val = body.substr(pe + 1, ve - pe - 1);
        if (!val.empty() && val[0] == '\n') val.erase(0, 1); if (!val.empty() && val.back() == '\n') val.pop_back();
        std::string type; if (props && props->find(key)) type = props->at(key).at("type").str();
        J v;
        if (type == "string") v = J(val);
        else { J parsed; const std::string tv = trim(val);
            if (tv == "True") v = J(true); else if (tv == "False") v = J(false); else if (tv == "None") v = J();
            else if (!tv.empty() && J::try_parse(tv, parsed) && (type.empty() ? !parsed.is_str() : true)) v = parsed; else v = J(val); }
        args.set(key, v); pos = ve + 12;
    }
    args_json = args.dump(); return true;
}

} // namespace q36

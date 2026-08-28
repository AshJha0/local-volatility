package com.quant.localvol;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal recursive-descent JSON parser for the flat golden-file schema
 * (objects, arrays, strings, numbers, booleans, null). Test-tree only.
 */
final class Json {

    private final String src;
    private int pos;

    private Json(String src) {
        this.src = src;
    }

    /** Parse a complete JSON document; trailing garbage is rejected. */
    static Object parse(String text) {
        Json p = new Json(text);
        p.skipWs();
        Object v = p.parseValue();
        p.skipWs();
        if (p.pos != text.length()) {
            throw new IllegalArgumentException("JSON: trailing characters at offset " + p.pos);
        }
        return v;
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> asObject(Object v) {
        return (Map<String, Object>) v;
    }

    @SuppressWarnings("unchecked")
    static List<Object> asArray(Object v) {
        return (List<Object>) v;
    }

    static double asDouble(Object v) {
        return ((Number) v).doubleValue();
    }

    private Object parseValue() {
        char c = peek();
        return switch (c) {
            case '{' -> parseObject();
            case '[' -> parseArray();
            case '"' -> parseString();
            case 't' -> parseLiteral("true", Boolean.TRUE);
            case 'f' -> parseLiteral("false", Boolean.FALSE);
            case 'n' -> parseLiteral("null", null);
            default -> parseNumber();
        };
    }

    private Map<String, Object> parseObject() {
        expect('{');
        Map<String, Object> obj = new LinkedHashMap<>();
        skipWs();
        if (peek() == '}') {
            pos++;
            return obj;
        }
        while (true) {
            skipWs();
            String key = parseString();
            skipWs();
            expect(':');
            skipWs();
            obj.put(key, parseValue());
            skipWs();
            char c = next();
            if (c == '}') {
                return obj;
            }
            if (c != ',') {
                throw new IllegalArgumentException("JSON: expected ',' or '}' at offset " + (pos - 1));
            }
        }
    }

    private List<Object> parseArray() {
        expect('[');
        List<Object> arr = new ArrayList<>();
        skipWs();
        if (peek() == ']') {
            pos++;
            return arr;
        }
        while (true) {
            skipWs();
            arr.add(parseValue());
            skipWs();
            char c = next();
            if (c == ']') {
                return arr;
            }
            if (c != ',') {
                throw new IllegalArgumentException("JSON: expected ',' or ']' at offset " + (pos - 1));
            }
        }
    }

    private String parseString() {
        expect('"');
        StringBuilder sb = new StringBuilder();
        while (true) {
            char c = next();
            if (c == '"') {
                return sb.toString();
            }
            if (c == '\\') {
                char e = next();
                switch (e) {
                    case '"' -> sb.append('"');
                    case '\\' -> sb.append('\\');
                    case '/' -> sb.append('/');
                    case 'b' -> sb.append('\b');
                    case 'f' -> sb.append('\f');
                    case 'n' -> sb.append('\n');
                    case 'r' -> sb.append('\r');
                    case 't' -> sb.append('\t');
                    case 'u' -> {
                        sb.append((char) Integer.parseInt(src.substring(pos, pos + 4), 16));
                        pos += 4;
                    }
                    default -> throw new IllegalArgumentException("JSON: bad escape \\" + e);
                }
            } else {
                sb.append(c);
            }
        }
    }

    private Object parseNumber() {
        int start = pos;
        while (pos < src.length() && "+-0123456789.eE".indexOf(src.charAt(pos)) >= 0) {
            pos++;
        }
        if (pos == start) {
            throw new IllegalArgumentException("JSON: unexpected character at offset " + pos);
        }
        return Double.parseDouble(src.substring(start, pos));
    }

    private Object parseLiteral(String lit, Object value) {
        if (!src.startsWith(lit, pos)) {
            throw new IllegalArgumentException("JSON: bad literal at offset " + pos);
        }
        pos += lit.length();
        return value;
    }

    private void skipWs() {
        while (pos < src.length() && Character.isWhitespace(src.charAt(pos))) {
            pos++;
        }
    }

    private char peek() {
        if (pos >= src.length()) {
            throw new IllegalArgumentException("JSON: unexpected end of input");
        }
        return src.charAt(pos);
    }

    private char next() {
        char c = peek();
        pos++;
        return c;
    }

    private void expect(char c) {
        if (next() != c) {
            throw new IllegalArgumentException("JSON: expected '" + c + "' at offset " + (pos - 1));
        }
    }
}

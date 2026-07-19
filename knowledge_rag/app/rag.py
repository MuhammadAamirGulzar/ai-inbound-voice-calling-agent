"""
RAG query layer: retrieval (Chroma, local) + generation (Qwen2.5-VL via
Ollama on the GPU machine).

Given a business_id and a question, this:
  1. Embeds the question and retrieves the most relevant chunks from that
     business's vector collection only (no cross-business leakage).
  2. Also pulls a handful of PUBLISHED menu items that best match the
     question (see app/vectorstore.py -> query_menu_items) so the model has
     real, priced items to reference/suggest — not just fuzzy text chunks.
     This is what fixes vague/garbled answers on questions like "what
     salads do you have" when the general Q&A chunks alone don't contain a
     clean answer: the menu-item search is precise (embedded per-item),
     while the general chunk search is fuzzy.
  3. Pulls the business's persona (tone) and upsell policy from its merged
     structured profile.
  4. Builds a grounded prompt — including recent conversation history, so a
     short follow-up like "haan" / "yes" after the assistant asks a
     question can be understood correctly — and asks the model to answer
     using ONLY the retrieved context, in the business's persona/tone, in
     Roman Urdu / Urdu script / English to match whatever the customer used.

Follow-up / conversation memory:
  This module itself is stateless per call — the caller (voice agent,
  Streamlit console, CLI) owns conversation state and re-sends the recent
  turns as `conversation_history` on every call. That's what lets a short
  reply like "yes" be understood as answering the assistant's own previous
  question, without this module needing a database of its own.

IMPORTANT HISTORY NOTE (why prices were coming out wrong/inconsistent):
  An earlier version of this prompt told the model to "say prices in words
  the way a person would say them" (e.g. "panch sau rupay"). That is
  backwards — asking a small local model to convert a digit into a spoken
  Urdu/Hindi number word is exactly how a wrong number slips in (e.g. the
  actual price is 295 but the model says "pachas" (50) or "char sau" (400)
  because it's improvising a word-form instead of reading the digit). The
  rule below now requires digits-only, copied straight from MENU ITEMS,
  and `_answer_has_spelled_out_price` enforces this in code as a backstop
  in case the prompt instruction alone isn't followed.
"""
import re

from . import ingest, llm_client, vectorstore

ANSWER_PROMPT_TEMPLATE = """You are the front-desk assistant for "{business_name}". Answer the customer's question using ONLY the context below. If the answer isn't in the context, say you don't have that information and offer to help with something else — do not make anything up.

Persona / tone to answer in: {tone}

LANGUAGE RULE (this is the most common mistake to avoid — follow it exactly):
Reply in the SAME language and SAME script the customer just used in "CUSTOMER MESSAGE" below — do not default to English and do not switch languages partway through your answer.
- If the customer wrote in Roman Urdu (Urdu words spelled with English/Latin letters, e.g. "aap ka menu kya hai", "delivery kitne baje tak hoti hai"), reply in Roman Urdu — Urdu words in Latin letters — NOT in the Urdu (Arabic) script and NOT in plain English. This business's customers are in Pakistan, so use Pakistani Urdu vocabulary specifically, not Hindi/Indian vocabulary — e.g. say "maloomat"/"tafseel" (not "jankari"), "shukriya" (not "dhanyavaad"), "abhi"/"filhaal" (not "abhi hi"), "paisay"/"rupay" (not "rupaye" in the Indian spelling-and-usage sense), "bilkul"/"zaroor" (not "avashya"). If in doubt whether a word is Hindi or Urdu, prefer the more common Pakistani everyday term over a Sanskrit-derived one.
- If the customer wrote in the Urdu script (اردو), reply in the Urdu script.
- If the customer wrote in plain English (e.g. "You have coffee", "do you have coffee?", "what time do you close"), reply in plain English — do NOT switch to Roman Urdu just because the business/persona is Pakistani. Example: customer message "You have coffee" -> a correct reply looks like "Yes, we have coffee — Cappuccino for 450 rupees, Latte for 500 rupees." in English, NOT a Roman Urdu sentence.
- If the customer mixed English and Roman Urdu in the same message, mirror that same natural mix back.
- The "Persona / tone to answer in" language preference above is only a fallback for when the message itself doesn't make the language clear (e.g. a single word, an emoji, a greeting like "hi"/"salam") — the customer's own wording always wins when it's clear.

GROUNDING / COHERENCE RULE (this is the second most common mistake — follow it exactly):
- Answer THIS message, not the previous one. Re-read CUSTOMER MESSAGE below and identify its own topic before looking at CONTEXT/MENU ITEMS. If CUSTOMER MESSAGE names its own item/category/topic, that topic — not whatever RECENT CONVERSATION was previously about — is what you answer. Only carry over the previous topic when CUSTOMER MESSAGE genuinely has no topic of its own (a bare "haan"/"yes"/"nahi", a pronoun like "uska"/"wo wala", or a short reply like "kitna hai" that only makes sense pointing at something just said).
- Every sentence you write must directly address what the customer actually asked, using only facts present in CONTEXT or MENU ITEMS below. Never produce a reply that is grammatically fine but topically unrelated to the question (e.g. answering a yes/no availability question with an unrelated sentence about payment, hours, or anything else not asked about) — and never answer with leftover content from an earlier question in this conversation.
- Before answering, check: does CONTEXT or MENU ITEMS actually contain something that answers THIS question? If yes, answer with that specific fact. If no, say plainly that you don't have that information yet (in the customer's language/script per the LANGUAGE RULE) and offer to help with something else — never fill the gap with a vague, generic, or tangential sentence just to have something to say.
- For a yes/no item-availability question (e.g. "do you have coffee", "you have coffee", "you have tea"), check MENU ITEMS: if a matching item is listed, answer "yes" and name the specific item(s)/price(s); if nothing matching is listed, answer "no"/"not currently" honestly — do not answer with an unrelated statement. A bare "yes, we have tea/coffee/X" with no item name or price attached is NEVER a complete answer if MENU ITEMS contains any matching item — always name the specific item(s) and price(s) from MENU ITEMS. This applies to every question that names or implies a category, not only ones phrased as "do you have" — if you are about to tell the customer something is available, and MENU ITEMS has priced items for it, name them.
- Whenever you name a menu item that has a price in MENU ITEMS, state that price in the same sentence. Never name an item and silently drop its price, even in a general "what's on your menu" style answer — if MENU ITEMS gives you a price for an item you're naming, say it.

COMPLETENESS RULE (this is the third most common mistake — a partial answer is a wrong answer here):
- If the customer asks for a LIST, ALL of something, "variety", or the price of "everything"/"sab" in a category (e.g. "kon konse soup hain aur keemat kya hai sab ki", "salad ki puri variety batao"), you MUST name EVERY matching item that appears in MENU ITEMS below, each with its own price — not just one representative item. Scan the entire MENU ITEMS list for every item in that category before answering; a reply naming only one item when three are listed in MENU ITEMS is an incomplete, incorrect answer.
- If MENU ITEMS only contains some of the category (retrieval may be imperfect), answer with everything that IS listed there and don't imply that's necessarily the complete menu unless CONTEXT confirms it is.
- Never substitute a single generic-sounding item for the full list just because it's easier to say — spoken answers can still name several items in one flowing sentence (see VOICE-CALL SPEAKING STYLE below).

GREETING-LEAK GUARD (important — read this even if CONTEXT looks like an opening script):
The CONTEXT below may contain a scripted call-opening or call-closing line (e.g. "Hi! Thank you for calling..."). That line is meant to be spoken ONCE at the very start/end of a call — it is NOT the answer to whatever the customer just asked. Unless the customer's message is itself a first-turn greeting ("hi", "hello", "salam") with RECENT CONVERSATION empty, do NOT open your reply by reciting that script and do NOT pad an unrelated answer with it — answer their actual question directly, using only the parts of CONTEXT/MENU ITEMS that are actually relevant to it. If neither CONTEXT nor MENU ITEMS actually contain the answer, say so plainly instead of talking around it.

NO RAW ECHO RULE (critical — this is a live spoken call, not a text lookup):
- CONTEXT may contain raw source material — old FAQ-style Q&A pairs, a price written as "RS. 245/295", the customer's own question repeated back, or other text lifted straight from an uploaded document. That raw text is a SOURCE for facts, never something to copy into your reply as-is.
- Never open your answer by repeating the customer's own question back to them (e.g. do not start with something equivalent to "قیمت کافی کا ہے؟" when the customer just asked that). Just answer it.
- Never output a bare price fragment on its own line (e.g. "RS. 245/295"). Always weave the price into one natural spoken sentence, in the customer's language/script, using the digit + "rupay"/"rupees" format from VOICE-CALL SPEAKING STYLE below — never the "RS."/"Rs."/"PKR"/"₨" symbol form, in ANY language or script, including Urdu script.
- Your entire answer must read as something a person would actually say out loud on a phone call — one or more complete, natural sentences — never a copy-pasted data fragment, label, or Q&A transcript line.

CLARITY RULE (keep every reply easy to follow out loud):
- If the customer's question can be answered yes/no or with one direct fact (e.g. "do you have coffee?", "are you open now?", "salad ki variety kya hai?"), START with that direct answer in one short sentence, THEN add relevant detail. Do not open by asking a question of your own — a caller who just asked something should get answered first.
- Ask AT MOST one question back per reply, only when you genuinely need it (e.g. to narrow down a variant/size), and phrase it as a single simple question — never stack two questions together.
- Keep sentences short and grammatically simple, especially when replying in Roman Urdu — prefer clear, simple phrasing over a long run-on sentence.

ORDER FOLLOW-UP RULE (this is a live call — keep the flow natural and one step at a time):
- If the customer says they want to order / place an order, but hasn't named a specific item yet (e.g. "I want to order", "mujhe order karna hai", "can I order something"), do NOT immediately list or push items. Instead, briefly confirm you can help, then ask a single short question: whether they'd like you to suggest something. Keep it to one or two short sentences.
- The "give a suggestion" behavior below applies ONLY when the assistant's OWN immediately-preceding message in RECENT CONVERSATION was literally that generic suggestion-offer question (e.g. "would you like me to suggest something?" / "kya mein apko kuch suggest kar doon?"). Check the actual text of your last message before treating "haan"/"yes" as triggering this — do not treat every bare "haan"/"yes" as "give a suggestion" by default.
- Only if RECENT CONVERSATION shows YOU just asked that specific suggestion-offer question, AND the customer's new message is an affirmative ("yes", "haan ji", "sure", "go ahead", etc.) — THEN give one concrete, proper suggestion: 1-3 specific items by name with their price, pulled ONLY from MENU ITEMS below (never invent an item or price). Follow the business's upsell/suggestion policy if one is given below. Keep it short and natural for speech, and end by asking if that works for them or if they'd like something else.
- If the customer's "haan"/"yes"/short affirmative comes after a DIFFERENT kind of question from you (e.g. you asked "which variant/size?", or asked to confirm an order, or asked anything else that wasn't the generic suggestion offer), resolve it against THAT actual question instead — per the FOLLOW-UP RULE below — never fall back to giving an unrelated menu suggestion just because the message happens to be "haan". A bare "haan" answering "which variant?" is ambiguous on its own; in that case briefly ask the customer to confirm which specific variant/size they'd like, rather than guessing or changing the subject.
- If the customer already named a specific item/category AND stated they want to order it in the SAME message (e.g. "mujhe espresso double shot order karni hai", "mujhe ek cappuccino chahiye"), do NOT re-ask a generic "would you like to order something?" question — they already answered that. Instead CONFIRM the order directly: name the exact item and its price from MENU ITEMS, then ask only what's still genuinely unknown (e.g. quantity, or a size/variant if MENU ITEMS lists more than one variant for that item) — or if nothing further is needed, confirm it's noted and ask if they'd like anything else.
- If the customer declines a suggestion ("no thanks", "nahi"), acknowledge briefly and ask what they'd like instead, without repeating the suggestion.
- Never suggest or state a price for anything not present in MENU ITEMS below.

IDENTITY RULE (for questions about the assistant itself, e.g. "apka naam kiya hai" / "who am I speaking with" / "what's your name"):
- Answer with ONLY your own name (given in the "Persona / tone to answer in" line above, if a name was set there), in one short, natural sentence. Do not tack on an unrelated follow-up question about the menu or anything else the customer didn't ask about — that violates the GROUNDING/COHERENCE RULE above. A simple "haan ji, mera naam Zara hai." (or the equivalent in the customer's language) is the complete, correct answer on its own. Only add a second sentence ("Aap ko kis cheez mein madad chahiye?" / "How can I help you today?") if it's a generic, non-topic-specific offer to help — never a specific unrelated question.

FOLLOW-UP RULE (use RECENT CONVERSATION to understand context, not just the literal words of this message):
- RECENT CONVERSATION below is the real conversation so far, oldest to newest. Read it before answering — a short message like "haan", "kitna hai", "aur kuch", or "ok" only makes sense in light of what YOU (the assistant) or the customer said just before it. Resolve pronouns/ellipsis ("uska price?", "vo wala") against the most recent relevant item discussed.
- Never ask the customer to repeat something they already told you earlier in RECENT CONVERSATION (e.g. their order, a preference, a name) — use what's already there.
- "ANYTHING ELSE / WHAT MORE" DEDUPE RULE: if the customer is asking for MORE options in a category you already listed (e.g. "aur kya hai", "starters mein aur kya hai", "kuch aur", "what else do you have") — check the ALREADY MENTIONED list below (if provided) and MENU ITEMS below for this same category. Do NOT repeat, word-for-word or near-word-for-word, any sentence naming items that are already in the ALREADY MENTIONED list — that reads as a broken/looping bot to the customer. If MENU ITEMS contains no items beyond what's already in ALREADY MENTIONED, say so plainly and briefly (e.g. "Bas yehi do hain filhaal, aur koi starter nahi hai." / "That's actually everything we have in starters right now.") — do NOT restate the earlier list as if it were new information. If MENU ITEMS DOES contain something beyond ALREADY MENTIONED, name ONLY the new item(s), not the ones already mentioned.

VOICE-CALL SPEAKING STYLE (this is read aloud by text-to-speech, not read on screen):
- PRICE ACCURACY (critical — read this carefully): state every price EXACTLY as the number that appears in MENU ITEMS below — same digits, nothing added or dropped. Say it as "295 rupay" or "295 rupees", NOT as a written symbol ("Rs.", "PKR", "₨"). Do NOT attempt to spell the number out in Urdu/Hindi words (e.g. do not write "pachas", "char sau", "hazar", "sattar") — that is where wrong prices come from, because guessing at a word-form of a number is how a different number slips in by mistake. A plain digit followed by "rupay"/"rupees" is said correctly by text-to-speech on its own and is copied straight from MENU ITEMS, so it can't drift.
- If the SAME item's price appears again later in this conversation (RECENT CONVERSATION), re-check it against MENU ITEMS below for THIS turn and use that number — never reuse a price you or the customer mentioned earlier in the conversation without re-verifying it against MENU ITEMS, and never state a price that isn't in MENU ITEMS at all. If a price you're about to say differs from a price you or the customer stated earlier in RECENT CONVERSATION for the same item, trust MENU ITEMS below over the earlier turn — the earlier number may itself have been wrong.
- Never use bullet points, numbered lists, markdown, asterisks, or emojis — this is spoken, not displayed. When listing 2-3 items in a sentence, just say them one after another naturally, e.g. "hamare paas X, Y aur Z hain."
- Keep it to 1-3 short spoken sentences per reply unless the customer asked for a full list.

WORKED EXAMPLES (this is the exact shape a correct reply looks like — study the structure, not the specific words):

Example 1 — availability question with variety, in Roman Urdu:
CUSTOMER MESSAGE: "apke pass soup hain?"
CORRECT ANSWER: "Jee bilkul, hamare paas soup ki achi variety hai — Chicken Corn Soup, Hot and Sour Soup aur Tomato Soup. Aap in mein se konsa order karna chahen ge?"
(Direct "yes" first, then the real items from MENU ITEMS with prices, then one simple question back — never open with a question, never pad with an unrelated greeting line.)

Example 2 — order follow-up flow across two turns, in Roman Urdu:
RECENT CONVERSATION: (empty)
CUSTOMER MESSAGE: "mujhe kuch order karna hai"
CORRECT ANSWER: "Jee zaroor! Kya mein apko kuch suggest kar doon, ya apke zehen mein pehle se kuch hai?"
--- next turn ---
RECENT CONVERSATION: "Assistant: Jee zaroor! Kya mein apko kuch suggest kar doon, ya apke zehen mein pehle se kuch hai?"
CUSTOMER MESSAGE: "haan"
CORRECT ANSWER: "Theek hai, hamare Chicken Karahi aur Zinger Burger kaafi pasand kiye jate hain. In mein se kaunsa apko pasand aaye ga?"
(The second reply correctly treats "haan" as answering the assistant's OWN question from RECENT CONVERSATION — it does not ask "suggest what?" or restart the conversation.)

Example 3 — plain English stays plain English, and a yes/no question still gets a concrete answer:
CUSTOMER MESSAGE: "do you have coffee?"
CORRECT ANSWER: "Yes, we have Cappuccino for 450 rupees and Latte for 500 rupees. Would you like to order one of these?"
(Same structure as Example 1, but the reply matches the customer's English — never switch to Roman Urdu just because the persona is Pakistani. Also note: "Yes, we have coffee" with no item/price named would NOT be a complete answer here — MENU ITEMS has specific items, so they must be named.)

Example 4 — "list everything in a category" question, in Roman Urdu (COMPLETENESS RULE):
RECENT CONVERSATION: "Customer: apke pass soups hain? / Assistant: Jee bilkul, hamare paas soup ki achi variety hai — Chicken Corn Soup, Hot and Sour Soup aur Tomato Soup. Aap in mein se konsa order karna chahen ge?"
CUSTOMER MESSAGE: "kon konse soup hain ar keemat kiya hai sb ki"
CORRECT ANSWER: "Jee zaroor, hamare paas Chicken Corn Soup 500 rupay ka hai, Hot and Sour Soup 450 rupay ka hai, aur Tomato Soup 400 rupay ka hai. Aap in mein se konsa order karna chahen ge?"
(Every soup that appears in MENU ITEMS is named with its own price — not just one soup with one price. This is still the SAME topic as the previous turn here, so history is used correctly, but every matching item is listed, not a partial answer.)

Example 5 — a genuinely new question right after an unrelated one — do NOT keep answering the old topic:
RECENT CONVERSATION: "Customer: you have coffee? / Assistant: Yes, we have Cappuccino for 450 rupees and Latte for 500 rupees."
CUSTOMER MESSAGE: "apke pass soups hain?"
CORRECT ANSWER: "Jee bilkul, hamare paas soup ki achi variety hai — Chicken Corn Soup, Hot and Sour Soup aur Tomato Soup. Aap in mein se konsa order karna chahen ge?"
(The new message names its own topic, "soup" — the reply is entirely about soup, with zero leftover mention of coffee, even though coffee was just discussed.)

Example 6 — identity question (IDENTITY RULE) — answer only what was asked, no unrelated follow-up:
CUSTOMER MESSAGE: "apka naam kiya hai"
CORRECT ANSWER: "Jee, mera naam Zara hai."
(Nothing else is added — no unrelated question about the menu. If a second sentence is added at all, it must be a generic offer to help, e.g. "Aap ki kya madad kar sakti hoon?" — never a specific unrelated question.)

Example 7 — "what else" follow-up in the same category (DEDUPE RULE) — do NOT repeat the same sentence verbatim:
RECENT CONVERSATION: "Customer: starters mein se kuch suggest kar dein / Assistant: Jee bilkul! Hamare paas Dynamite Chicken aur Dynamite Prawn (5PC) starters ki achi variety hai. Aap kuch order karna chahen ge?"
ALREADY MENTIONED IN THIS CONVERSATION: Dynamite Chicken, Dynamite Prawn (5PC)
CUSTOMER MESSAGE: "starters mein aur kya hai"
MENU ITEMS: only Dynamite Chicken and Dynamite Prawn (5PC) — no other starter items.
CORRECT ANSWER: "Bas yehi do hain filhaal — Dynamite Chicken aur Dynamite Prawn. Koi aur starter abhi available nahi hai. In mein se kaunsa order karna chahen ge?"
(WRONG shape to avoid: repeating "Hamare paas Dynamite Chicken aur Dynamite Prawn (5PC) starters ki achi variety hai" again word-for-word, as if it's new information — that reads as a broken/looping bot to the customer. Since everything in MENU ITEMS is already in ALREADY MENTIONED, the correct reply plainly says that's everything, in different words.)

Example 8 — "haan" after a NON-suggestion question — do NOT default to giving an unrelated menu suggestion:
RECENT CONVERSATION: "Customer: mujhe ek latte order karni hai / Assistant: Jee bilkul! Hamare paas Caffè Latte 645 rupay ka hai. Aap konsa variant order karna chahen ge?"
CUSTOMER MESSAGE: "haan"
WRONG ANSWER (do not do this): "Jee bilkul! Hamare paas starters ki achi variety hai — Salsa Chicken Tacos..." (this ignores what was actually asked — "which variant?" — and jumps to an unrelated suggestion just because the message is "haan")
CORRECT ANSWER: "Theek hai — kaunsa variant lena chahen ge, regular ya large?" (or, if MENU ITEMS only shows one variant/size for Latte, treat "haan" as confirming that one and proceed: "Theek hai, ek Caffè Latte 645 rupay wala note kar liya. Aur kuch chahiye?")
(The assistant's last question here was about a VARIANT, not the generic "would you like a suggestion?" offer — so "haan" must be resolved against the variant question per the ORDER FOLLOW-UP RULE, never treated as triggering a fresh, unrelated suggestion.)

Example 9 — do not echo raw context (NO RAW ECHO RULE), in Urdu script:
CONTEXT contains a raw FAQ line: "قیمت کافی کا ہے؟ RS. 245/295 — Espresso Single Shot RS. 245 Espresso Double Shot RS. 295"
CUSTOMER MESSAGE: "قیمت کیا ہے کافی کی؟"
WRONG ANSWER (do not do this): "قیمت کافی کا ہے؟ RS. 245/295 Espresso Single Shot RS. 245 Espresso Double Shot RS. 295" (this just copies the raw source text and repeats the question back — not a real sentence, and uses the forbidden "RS." symbol)
CORRECT ANSWER: "جی بالکل، ہمارے پاس Espresso Single Shot 245 روپے کا ہے اور Espresso Double Shot 295 روپے کا ہے۔ آپ کونسا آرڈر کرنا چاہیں گے؟"
(One natural spoken sentence in Urdu script, digits only for the prices, no symbol, no echoed question, no raw copy-paste.)

Upsell / suggestion policy: {upsell_strategy}

CONTEXT:
{context}

MENU ITEMS (only real, published items — use these for any suggestion or price; if empty, say you don't have menu details loaded yet):
{menu_context}

ALREADY MENTIONED IN THIS CONVERSATION (items you or the customer already named earlier in this call — see the DEDUPE RULE above; do not repeat these as if they were new, unless the customer is explicitly asking about them again):
{already_mentioned}

RECENT CONVERSATION (oldest to newest, may be empty for the first message):
{history}

CUSTOMER MESSAGE: {question}

Answer (remember: match the customer's language/script exactly per the LANGUAGE RULE, use RECENT CONVERSATION per the FOLLOW-UP RULE, follow the ORDER FOLLOW-UP RULE above, and never repeat items already listed in ALREADY MENTIONED as if they were new):"""


def _format_history(conversation_history: list) -> str:
    if not conversation_history:
        return "(none — this is the first message)"
    lines = []
    for turn in conversation_history[-8:]:  # keep the prompt small; recent turns are what matter most
        role = (turn.get("role") or "").strip().lower()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        speaker = "Assistant" if role == "assistant" else "Customer"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines) if lines else "(none — this is the first message)"


RETRIEVAL_QUERY_PROMPT = """Convert the customer message below into a short English search query naming the food/drink item, category, or topic being asked about. This query is ONLY used to search an English menu database — it is never shown to the customer, so do not answer their question, do not add greetings or explanations, do not restate the rules.

Your entire reply must be JUST the search query — 1 to 4 English words — and nothing else. Not a sentence. Not "Sure, here is the query:". Not quotes around it. Just the words.

TOPIC RULE (read this first — this is the most common mistake to avoid):
The CUSTOMER MESSAGE almost always names its OWN topic (an item, a category, a word like "soup"/"delivery"/"menu"). When it does, build the query from THAT topic — completely ignore whatever RECENT CONVERSATION was about, even if the two topics are related (e.g. both are food, or both are about soup). A new, self-contained question is NOT a follow-up just because it comes right after another question. Do not let a previous answer "stick" — every new message gets re-evaluated on its own words first.
Only fall back to RECENT CONVERSATION's topic when the CUSTOMER MESSAGE has NO topic word of its own at all — it's a bare pronoun, a single yes/no ("haan", "yes", "nahi"), a vague reference ("uska", "wo wala", "iska"), or a short reply like "kitna hai" / "aur kuch" that only makes sense pointing at something already said.

COMPLETENESS RULE:
If the customer is asking for a LIST, ALL options, prices of everything in a category, or "variety" ("sab", "sari", "poori list", "konse konse", "variety", "kya kya hai", "all"), output the CATEGORY word only (e.g. "soup", "salad") — never narrow it down to one specific item name. Narrowing a "give me everything" question to a single item is what causes an incomplete answer later.

Examples (input -> output, nothing but the arrow's right side is ever written):
"apke pass soups hain" -> soup
"aik chicken sadwich kr dein" -> chicken sandwich
"apke pass starter mn kiya hai" -> starters
"salad ki variety kya hai" -> salad
"you have coffee" -> coffee
"delivery kitne baje tak hoti hai" -> delivery timings
"kon konse soup hain ar keemat kiya hai sb ki" -> soup            (category word, NOT one specific soup — this is a "list everything" question)
"apka naam kiya hai" -> assistant name
"aap ka menu kya hai" -> menu
"haan" (with RECENT CONVERSATION showing you just offered coffee) -> coffee
"kitna hai" (with RECENT CONVERSATION about a Cappuccino) -> cappuccino price
"aur soup?" (with RECENT CONVERSATION about coffee) -> soup        (message names its own new topic "soup" — do NOT keep searching coffee just because it was the previous topic)
"delivery kab tak hai" (with RECENT CONVERSATION about coffee) -> delivery timings   (brand-new self-contained topic — ignore the previous coffee topic entirely)

RECENT CONVERSATION (oldest to newest, may be empty):
{history}

CUSTOMER MESSAGE: {question}

Output (1-4 English words, nothing else):"""

# Small/local models frequently ignore "output only X" and prepend a lead-in
# anyway. Strip the common lead-in shapes before using the text as a search
# query, so a chatty translation doesn't dilute the embedding with noise
# words that have nothing to do with the actual item/category.
_TRANSLATION_PREAMBLE_PREFIXES = (
    "sure", "here is", "here's", "the query is", "query:", "answer:",
    "output:", "translation:", "search query:", "english search query:",
)


def _clean_translated_query(text: str) -> str:
    text = (text or "").strip().strip('"').strip("'").strip()
    # If it's multi-sentence/multi-line noise, keep only the first line —
    # the real query is almost always the first thing produced.
    text = text.splitlines()[0].strip() if text else text
    lowered = text.lower()
    for prefix in _TRANSLATION_PREAMBLE_PREFIXES:
        if lowered.startswith(prefix):
            # cut everything up to and including the first colon/dash after the lead-in
            for sep in (":", "-", "—"):
                if sep in text:
                    text = text.split(sep, 1)[1].strip()
                    break
            break
    return text.strip().strip('"').strip("'")


def _build_retrieval_query(question: str, conversation_history: list) -> str:
    """
    Embeddings here (see app/config.py -> EMBEDDING_MODEL_NAME) are produced
    by an English-only model. Menu items and general context chunks are in
    English, but customers frequently ask in Roman Urdu (e.g. "aik chicken
    sadwich kr dein") or Urdu script — an English-only embedder places that
    text far from the matching English menu text in vector space, so
    similarity search comes back empty/irrelevant even when the item exists.
    This is the actual cause of "I don't have that information" on
    perfectly answerable questions.

    Fix: translate/normalize the question into a short English query BEFORE
    embedding/searching. The final answer is still generated from the
    customer's original `question` (untouched) via ANSWER_PROMPT_TEMPLATE,
    so the reply still comes back in the customer's own language/script —
    this function only changes what gets searched, never what gets said.

    The small local model this runs on often doesn't follow "output only
    the query" cleanly, so the prompt above is few-shot + tightly
    constrained, and `_clean_translated_query` strips common lead-in noise
    ("Sure, here's the query: ...") as a second line of defense.

    Falls back to the raw question on any translation failure/empty output,
    so a model hiccup degrades retrieval quality instead of breaking the
    call — but that fallback is exactly the old broken behavior, so if
    answers are still wrong after this, the translation call itself is
    likely failing/timing out silently; that's worth checking in your
    Ollama/tunnel logs next.
    """
    prompt = RETRIEVAL_QUERY_PROMPT.format(
        history=_format_history(conversation_history or []),
        question=question,
    )
    try:
        raw = llm_client.chat_text(prompt, temperature=0.0, max_tokens=20)
        cleaned = _clean_translated_query(raw)
        return cleaned if cleaned else question
    except Exception:
        return question


def _dedupe_menu_items(matches: list) -> list:
    """Retrieval can return the same item name more than once — most often
    because the vector store still holds a stale duplicate entry from an
    earlier re-ingest of the menu (same item, old price, never cleaned up)
    sitting alongside the current one. Feeding the model two entries for
    the same item with two different prices is exactly how a customer gets
    told two different numbers for the same thing across turns — the model
    has no way to know which one is authoritative, so it just repeats
    whichever entry it read first.

    This keeps ONE entry per (name, category) — the first one returned,
    since vectorstore.query_menu_items already orders by relevance — so the
    prompt only ever sees a single, unambiguous price per item per call.

    This is a mitigation, not a fix for the root cause: if prices are still
    inconsistent across separate calls (not within one call) after this,
    the vector store itself has duplicate/stale embeddings for the same
    item and menu ingestion should upsert by a stable item ID instead of
    always inserting a new vector.
    """
    seen = set()
    deduped = []
    for m in matches:
        key = ((m.get("name") or "").strip().lower(), (m.get("category") or "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(m)
    return deduped


def _format_menu_items(matches: list) -> str:
    if not matches:
        return "(no menu items available)"
    lines = []
    for m in matches:
        name = m.get("name", "")
        category = m.get("category", "")
        price = m.get("price", "")
        variants = m.get("variants") or []
        line = f"- {name}"
        if category:
            line += f" ({category})"
        if price:
            line += f" — {price}"
        if variants:
            variant_str = ", ".join(f"{v.get('name')}: {v.get('price')}" for v in variants if v.get("name"))
            if variant_str:
                line += f" [{variant_str}]"
        lines.append(line)
    return "\n".join(lines)


def _extract_already_mentioned_items(conversation_history: list, menu_matches: list) -> list:
    """Best-effort: which of the currently-retrieved menu item names were
    already spoken by the assistant earlier in this conversation?

    This exists because the DEDUPE RULE in the prompt used to rely entirely
    on the model reading RECENT CONVERSATION prose and figuring out for
    itself what it already said — that's exactly what was failing (the
    model would just repeat its previous sentence verbatim instead of
    recognizing it as a repeat). Handing the model an explicit,
    pre-computed list removes the inference step it was getting wrong.

    Simple substring match on item name against the concatenated assistant
    turns — good enough for this purpose since menu item names are fairly
    distinctive strings, not common words.
    """
    if not conversation_history or not menu_matches:
        return []
    assistant_text = " ".join(
        (turn.get("content") or "")
        for turn in conversation_history
        if (turn.get("role") or "").strip().lower() == "assistant"
    ).lower()
    if not assistant_text.strip():
        return []
    mentioned = []
    for m in menu_matches:
        name = (m.get("name") or "").strip()
        if name and name.lower() in assistant_text and name not in mentioned:
            mentioned.append(name)
    return mentioned


def _format_already_mentioned(names: list) -> str:
    return ", ".join(names) if names else "(none yet)"


_URDU_SCRIPT_RE = re.compile(r"[\u0600-\u06FF]")

# Common Roman Urdu function words/markers — deliberately Pakistani-Urdu
# flavored (see LANGUAGE RULE above), used only as a cheap signal, not a
# full language classifier. Word-boundary matched, case-insensitive.
_ROMAN_URDU_MARKERS = {
    "hai", "hain", "ka", "ki", "ke", "kya", "kyun", "kaise", "kitna", "kitne",
    "aap", "ap", "apka", "apke", "apki", "mera", "mere", "meri", "hum",
    "hamare", "hamara", "chahiye", "chahen", "chahenge", "karna", "kro",
    "krna", "kr", "nahi", "nahin", "haan", "han", "ji", "jee", "acha",
    "theek", "thk", "bilkul", "zaroor", "shukriya", "paisay", "rupay",
    "wala", "wali", "kaunsa", "konsa", "abhi", "filhaal", "maloomat",
    "tafseel", "aur", "ya", "se", "mein", "main", "pe", "par", "toh", "to",
}


# Unambiguous Urdu-only tokens for the reverse check below (customer wrote
# English, did the reply wrongly switch to Roman Urdu?). Deliberately a
# SMALLER, stricter set than _ROMAN_URDU_MARKERS above: several tokens in
# that broader set ("to", "main", "par", "se", "ya", "ka", "ki") are also
# ordinary English words ("to order", "the main dish", "par excellence",
# "ya" as a name) and would false-positive constantly if used here. This
# set only contains words that essentially never appear in genuine English
# text, so a hit here is a reliable signal, not a coincidence.
_UNAMBIGUOUS_ROMAN_URDU_MARKERS = {
    "hai", "hain", "kya", "kitna", "kitne", "aap", "apka", "apke", "apki",
    "mera", "mere", "meri", "hamare", "hamara", "chahiye", "chahen",
    "chahenge", "nahi", "nahin", "haan", "jee", "theek", "bilkul", "zaroor",
    "shukriya", "paisay", "rupay", "kaunsa", "konsa", "filhaal", "maloomat",
    "tafseel",
}

# Spelled-out Urdu/Hindi number words. If any of these show up in a reply,
# the model has spelled out a price in words instead of copying the digit
# straight from MENU ITEMS — this is precisely how a wrong/rounded price
# slips in ("pachas" instead of the actual 295, "char sau" instead of the
# actual 295), even though the prompt explicitly forbids it. Enforced here
# in code rather than left to the prompt alone, the same way the language
# rule is enforced below.
_SPELLED_OUT_NUMBER_WORDS = {
    "ek", "do", "teen", "char", "paanch", "chhe", "che", "saat", "aath",
    "nau", "dus", "das", "gyarah", "barah", "terah", "chodah", "pandrah",
    "solah", "satrah", "atharah", "unnees", "bees", "tees", "chalis",
    "chaalis", "pachas", "sath", "saath", "sattar", "assi", "nabbe",
    "sau", "hazar", "hazaar", "lakh",
}

# Written currency-symbol forms ("RS. 245", "Rs 245", "PKR 245", "₨245").
# The prompt explicitly forbids these in favor of "245 rupay" (see
# VOICE-CALL SPEAKING STYLE / NO RAW ECHO RULE) because this is a voice
# call — a written symbol either gets read aloud oddly by TTS or, worse,
# is a sign the model copy-pasted a raw price fragment straight out of
# CONTEXT instead of composing a spoken sentence (see the Urdu-script
# "RS. 245/295" raw-dump bug). Matches regardless of script/language the
# rest of the answer is in, since the symbol itself is never correct.
_CURRENCY_SYMBOL_RE = re.compile(r"(?i)\brs\.?\s*\d|\bpkr\b|₨")


def _detect_customer_language(text: str) -> str:
    """Cheap heuristic, not a real language classifier — good enough to
    catch the common failure (model answers in English/wrong script when
    the customer clearly didn't). Returns 'urdu_script' | 'roman_urdu' |
    'english'."""
    if _URDU_SCRIPT_RE.search(text or ""):
        return "urdu_script"
    words = re.findall(r"[a-zA-Z]+", (text or "").lower())
    marker_hits = sum(1 for w in words if w in _ROMAN_URDU_MARKERS)
    if marker_hits >= 1:
        return "roman_urdu"
    return "english"


def _reply_matches_language(answer: str, expected: str) -> bool:
    """Does the generated answer actually look like it's in the language
    the customer used? Deliberately lenient (menu item names are often
    English/branded even in a Roman Urdu reply, e.g. 'Zinger Burger') —
    this only needs to catch the clear failure case, not police every
    sentence."""
    if expected == "urdu_script":
        return bool(_URDU_SCRIPT_RE.search(answer or ""))
    if expected == "roman_urdu":
        if _URDU_SCRIPT_RE.search(answer or ""):
            return False  # wrong script entirely
        words = re.findall(r"[a-zA-Z]+", (answer or "").lower())
        marker_hits = sum(1 for w in words if w in _ROMAN_URDU_MARKERS)
        return marker_hits >= 1
    # expected == "english": checked with the STRICT marker set above
    # (>=2 hits) so ordinary English sentences containing one incidental
    # overlap word don't trigger a false correction.
    if _URDU_SCRIPT_RE.search(answer or ""):
        return False
    words = re.findall(r"[a-zA-Z]+", (answer or "").lower())
    marker_hits = sum(1 for w in words if w in _UNAMBIGUOUS_ROMAN_URDU_MARKERS)
    return marker_hits < 2


def _answer_has_spelled_out_price(answer: str) -> bool:
    """True if the answer looks like it spelled a price out in Urdu/Hindi
    number words instead of using digits copied from MENU ITEMS. This is a
    heuristic (a couple of these words, like 'do'/'char', can rarely appear
    as ordinary short words) but in practice a hit here is almost always a
    spelled-out number, and the cost of an occasional unnecessary
    regeneration is far lower than the cost of a wrong price reaching a
    customer on a live call."""
    words = re.findall(r"[a-zA-Z]+", (answer or "").lower())
    return any(w in _SPELLED_OUT_NUMBER_WORDS for w in words)


def _answer_has_currency_symbol(answer: str) -> bool:
    """True if the answer used 'RS.'/'PKR'/'₨' instead of a plain digit +
    rupay/rupees. A written currency symbol on a voice call is a real bug
    on its own, and it's also a strong signal the model copy-pasted a raw
    price fragment out of CONTEXT instead of composing a sentence — see
    the Urdu-script raw-dump bug this was added to catch."""
    return bool(_CURRENCY_SYMBOL_RE.search(answer or ""))


def _price_format_ok(answer: str) -> bool:
    return not _answer_has_spelled_out_price(answer) and not _answer_has_currency_symbol(answer)


def _answer_missing_available_prices(answer: str, menu_matches: list) -> bool:
    """True if MENU ITEMS actually has priced items, but the generated
    answer names/implies items without stating a single digit anywhere —
    e.g. "Yes, we have tea options available" with no item or price named.
    This is a coarse heuristic (it doesn't check that the RIGHT price was
    used, only that SOME digit appears), but it's exactly what catches a
    bare "yes, we have X" answer that dodges naming anything concrete, per
    the GROUNDING/COMPLETENESS rules above. Skipped when MENU ITEMS is
    empty or has no priced items at all, since then there's genuinely
    nothing to name (e.g. a pure hours/delivery/identity question)."""
    has_priced_item = any((m.get("price") or (m.get("variants") or [])) for m in (menu_matches or []))
    if not has_priced_item:
        return False
    return not re.search(r"\d", answer or "")


def _build_correction_note(lang_ok: bool, expected_lang: str, price_format_ok: bool, has_currency_symbol: bool, prices_missing: bool) -> str:
    """Builds one combined corrective instruction covering whichever checks
    failed (language, price format, currency symbol, or missing prices),
    for a single regeneration pass — small local models occasionally
    ignore these rules despite them being spelled out in the main prompt,
    so this is the actual enforcement, not just a hope that the
    instruction was followed."""
    notes = []
    if not lang_ok:
        notes.append({
            "roman_urdu": (
                "your previous answer was not in Roman Urdu. Re-answer in Roman Urdu "
                "only (Urdu words spelled in Latin/English letters, Pakistani "
                "vocabulary) — not Urdu script, not plain English."
            ),
            "urdu_script": (
                "your previous answer was not in the Urdu script. Re-answer written "
                "in the Urdu (اردو) script only, as one natural spoken sentence — do "
                "not repeat the customer's question back and do not copy raw text "
                "from CONTEXT."
            ),
            "english": (
                "your previous answer switched into Roman Urdu, but the customer "
                "wrote their message in plain English. Re-answer in plain English "
                "only — do not use Roman Urdu or Urdu-script words, even if the "
                "business's persona is Pakistani."
            ),
        }.get(expected_lang, ""))
    if not price_format_ok:
        notes.append(
            "your previous answer spelled a price out in Urdu/Hindi words (e.g. "
            "'pachas', 'char sau', 'hazar') instead of using the exact digits from "
            "MENU ITEMS. Re-answer using ONLY digit prices copied exactly from MENU "
            "ITEMS, e.g. '295 rupay' — never a word-form number, and never a "
            "guessed or rounded number."
        )
    if has_currency_symbol:
        notes.append(
            "your previous answer used a written currency symbol ('RS.', 'PKR', "
            "'₨') instead of a spoken digit + rupay/rupees. Re-answer as one "
            "natural spoken sentence with the price as a plain digit followed by "
            "'rupay'/'rupees' (e.g. '295 rupay'), never the symbol form, and never "
            "a raw copy-pasted price fragment on its own."
        )
    if prices_missing:
        notes.append(
            "your previous answer said items/options are available but did not "
            "name any specific item or its price, even though MENU ITEMS below has "
            "priced items that match. Re-answer naming the specific item(s) and "
            "their exact price(s) from MENU ITEMS — never a bare 'yes, we have "
            "that' with nothing concrete attached."
        )
    notes = [n for n in notes if n]
    if not notes:
        return ""
    return (
        "\n\nCORRECTION NEEDED: "
        + " ALSO: ".join(notes)
        + " Re-answer the SAME question with these fixes applied."
    )


def _regenerate_with_correction(prompt: str, correction: str) -> str:
    if not correction:
        return ""
    try:
        return llm_client.chat_text(prompt + correction, temperature=0.15, max_tokens=1024)
    except Exception:
        return ""


def answer_question(business_id: str, question: str, top_k: int = 8, conversation_history: list = None) -> dict:
    conversation_history = conversation_history or []
    retrieval_query = _build_retrieval_query(question, conversation_history)
    # TEMP DEBUG — remove once retrieval is confirmed reliable. If this
    # prints the same Roman Urdu text as the original question (unchanged),
    # the translation call is failing/timing out and silently falling back
    # — check the Ollama/tunnel connection, not the prompt, in that case.
    print(f"[rag] question={question!r} -> retrieval_query={retrieval_query!r}")
    results = vectorstore.query(business_id, retrieval_query, top_k=top_k)
    # top_k=12 here (not the old 6): a "list everything in this category"
    # question (see COMPLETENESS RULE) needs enough candidates that every
    # matching menu item actually makes it into MENU ITEMS below, not just
    # the closest one or two. _MAX_RELEVANT_DISTANCE in vectorstore.py
    # still filters out anything that isn't a real match, so this doesn't
    # add noise for a single-item lookup — it only helps when there are
    # more matching items to find.
    menu_matches = vectorstore.query_menu_items(business_id, retrieval_query, top_k=12)
    # De-duplicate before anything downstream ever sees these — see
    # _dedupe_menu_items docstring for why (stale duplicate vectors ->
    # same item, two different prices, in the same prompt).
    menu_matches = _dedupe_menu_items(menu_matches)

    profile = ingest.get_merged_business_profile(business_id)
    
    profile_context_lines = []
    if profile:
        details = profile.get("details") or {}
        policies = profile.get("policies") or {}
        
        if details.get("address"):
            profile_context_lines.append(f"Address: {details['address']}")
        if details.get("phone"):
            profile_context_lines.append(f"Phone Number: {details['phone']}")
        if details.get("hours"):
            profile_context_lines.append(f"Hours: {details['hours']}")
        if details.get("avg_preparation_time"):
            profile_context_lines.append(f"Average Food Prep Time: {details['avg_preparation_time']}")
        if details.get("delivery_info"):
            profile_context_lines.append(f"Delivery Details: {details['delivery_info']}")
        if details.get("delivery_areas"):
            profile_context_lines.append(f"Delivery Areas: {', '.join(details['delivery_areas'])}")
            
        if policies.get("upsell_strategy"):
            profile_context_lines.append(f"Upsell Policy: {policies['upsell_strategy']}")
        if policies.get("out_of_stock_protocol"):
            profile_context_lines.append(f"Out of Stock Protocol: {policies['out_of_stock_protocol']}")
        if policies.get("escalation_protocol"):
            profile_context_lines.append(f"Escalation Policy: {policies['escalation_protocol']}")

    if not results and not menu_matches and not profile_context_lines:
        return {
            "answer": "I don't have any information ingested for this business yet — please upload its menu, persona, or details first.",
            "sources": [],
        }

    context_chunks = []
    if profile_context_lines:
        context_chunks.append("BUSINESS DETAILS & POLICIES:\n" + "\n".join(profile_context_lines))
    if results:
        context_chunks.append("BUSINESS KNOWLEDGE:\n" + "\n\n---\n\n".join(doc for doc, _ in results))
        
    context = "\n\n".join(context_chunks) if context_chunks else "(no general Q&A context matched)"

    business_name = profile.get("business_name") or business_id if profile else business_id
    persona = profile.get("persona") or {} if profile else {}
    tone = persona.get("tone") or "friendly and helpful"
    if persona.get("agent_name"):
        tone = f"{tone}. Your name is {persona['agent_name']} — if the customer asks your name, answer with just that name in one short sentence (see IDENTITY RULE)."

    policies = profile.get("policies") or {}
    upsell_strategy = policies.get("upsell_strategy") or "No specific policy given — suggest tastefully and only when asked, never pushy."

    already_mentioned = _extract_already_mentioned_items(conversation_history, menu_matches)

    prompt = ANSWER_PROMPT_TEMPLATE.format(
        business_name=business_name,
        tone=tone,
        context=context,
        menu_context=_format_menu_items(menu_matches),
        already_mentioned=_format_already_mentioned(already_mentioned),
        history=_format_history(conversation_history),
        upsell_strategy=upsell_strategy,
        question=question,
    )

    answer = llm_client.chat_text(prompt, temperature=0.15, max_tokens=1024)

    # Enforcement layer for the LANGUAGE RULE and PRICE ACCURACY rule:
    # don't just hope the model followed them — check, and if it clearly
    # didn't, force one corrective regeneration covering whichever checks
    # failed. This is what makes Roman Urdu / Urdu-script matching and
    # digit-only prices reliable on a small local model instead of
    # "usually works."
    expected_lang = _detect_customer_language(question)
    lang_ok = _reply_matches_language(answer, expected_lang)
    price_ok = not _answer_has_spelled_out_price(answer)
    has_currency_symbol = _answer_has_currency_symbol(answer)
    prices_missing = _answer_missing_available_prices(answer, menu_matches)
    
    if not lang_ok or not price_ok or has_currency_symbol or prices_missing:
        correction = _build_correction_note(lang_ok, expected_lang, price_ok, has_currency_symbol, prices_missing)
        retried = _regenerate_with_correction(prompt, correction)
        if retried:
            retried_lang_ok = _reply_matches_language(retried, expected_lang)
            retried_price_ok = not _answer_has_spelled_out_price(retried)
            retried_currency = _answer_has_currency_symbol(retried)
            retried_missing = _answer_missing_available_prices(retried, menu_matches)
            if retried_lang_ok and retried_price_ok and not retried_currency and not retried_missing:
                answer = retried

    return {
        "answer": answer,
        "sources": sorted(set(meta.get("source_file", "") for _, meta in results)),
    }

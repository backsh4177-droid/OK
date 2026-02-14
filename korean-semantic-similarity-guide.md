# 한국어 단어 근접성/유사도 측정 빠른 가이드

쿠모쿤의 요구사항(“한국어 단어를 넣으면 유사도가 수치로 나오고, AI가 어떤 맥락으로 묶는지 감을 잡는 지표”)에 맞춰서 바로 써먹을 수 있는 선택지만 정리했습니다.

## 1) 웹에서 바로 입력 → 결과 확인

### A. Korean Word2Vec
- 한국어 형태소 기반 Word2Vec 데모.
- 비슷한 맥락 단어(근접 이웃) 확인에 적합.
- 코퍼스 정보가 공개되어 있어 결과 해석에 도움.
- 링크: https://word2vec.kr/

### B. ETRI 기술체험 플랫폼 — 어휘 간 유사도 분석
- 단어 2개를 입력해 어휘 의미망 기반 유사도/거리 분석 가능.
- “유의어 사전”이 아니라 의미 관계를 정량적으로 보기 좋음.
- 링크: https://epretx.etri.re.kr/api/demo/WordRel

## 2) Gemini/Claude 실사용 구조와 가장 유사한 방식

### A. Gemini: 임베딩 + 코사인 유사도
- 모델: `gemini-embedding-001`
- 권장 task type: `SEMANTIC_SIMILARITY`
- 출력 임베딩 간 코사인 유사도(-1~1) 계산

```python
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity

client = genai.Client()

texts = ["단어/문장 A", "단어/문장 B"]
res = client.models.embed_content(
    model="gemini-embedding-001",
    contents=texts,
    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
)

sim = cosine_similarity([res.embeddings[0].values], [res.embeddings[1].values])[0][0]
print(sim)
```

팁: 단어 단독보다 짧은 문장으로 맥락을 붙여 넣어야 다의어 흔들림이 줄어듭니다.

### B. Claude: 자체 임베딩 미제공
- Anthropic은 자체 임베딩 모델을 제공하지 않음.
- 일반적으로 Claude(생성) + 외부 임베딩 서비스(예: Voyage AI) 조합 사용.
- 링크: https://platform.claude.com/docs/en/build-with-claude/embeddings

## 3) 사전/용어 API로 교차검증

임베딩은 “같이 등장하는 맥락”을 반영하므로 반의어도 가까워질 수 있습니다. 따라서 의미(정의/용례/대체어)는 사전 데이터로 함께 확인하는 것이 안전합니다.

- 우리말샘: https://opendict.korean.go.kr/
- 온용어 오픈 API: https://kli.korean.go.kr/term/bbs/indexOpenApiInfo.do

## 4) “AI가 어떤 맥락으로 예측하나”를 수치로 보는 실전 방법

1. 같은 핵심 단어로 문장 3종 생성
   - 중립 문장
   - 긍정 맥락 문장
   - 부정 맥락 문장
2. 각 문장 임베딩을 만든 뒤 상호 유사도 계산
3. 어떤 의미축(다의성/감성축)으로 분기되는지 확인
4. 원하는 의미를 대표하는 anchor 문장을 기준으로 후보 문장/단어를 유사도 순 정렬

이 방식은 검색/분류/RAG 전처리에서 바로 활용 가능합니다.

## 참고 링크
- https://word2vec.kr/
- https://epretx.etri.re.kr/api/demo/WordRel
- https://ai.google.dev/gemini-api/docs/embeddings
- https://platform.claude.com/docs/en/build-with-claude/embeddings
- https://opendict.korean.go.kr/
- https://kli.korean.go.kr/term/bbs/indexOpenApiInfo.do

This is a supporting repo to the main repo of a research project. This repo analyzes data from hte proejct. I've pasted below the research paper draft that this project is based on, as well as some details / conversations from my PI about it. Me (Shreyas) and my colleague Deonna are working on visualizing the data. 

Paper Draft: 
\section{Experiments}\label{sec:experiments}

We present experiments in two parts: a single-agent, single-round baseline that characterizes fragility of existing fairness measurements (\S\ref{sec:exp-baseline}), and a multi-agent, multi-round evaluation that surfaces covert discrimination through behavioral dynamics (\S\ref{sec:exp-multiagent}).
All results are reported across three benchmarks described in \S\ref{sec:benchmarks}.

\paragraph{Models.} We evaluate a range of open-weight instruction-tuned models spanning several orders of magnitude in size:
\begin{itemize}
    \item \textbf{Small} (1B--4B): Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, Qwen3-4B-Instruct, Phi-3-mini, Phi-3.5-mini, Phi-4-mini, Ministral-3B.
    \item \textbf{Medium} (7B--9B): Llama-3.1-8B-Instruct, Gemma-2-9B-it, Ministral-8B, Mistral-7B-v0.3.
    \item \textbf{Large} (27B--70B): Gemma-2-27B-it, Llama-3.1-70B-Instruct, Llama-3.3-70B-Instruct, Ministral-14B.
\end{itemize}
\st{TODO: trim to actual models used once all runs complete; above is the planned set.}
\zt{TODO: provide a new list prioritizing the most recent models and ignore older models, especially when they are from the same model family}
All models are served via vLLM with continuous batching on local GPU infrastructure.
Unless stated otherwise, we use temperature $\tau = 0.7$ and a maximum context length of 2{,}048 tokens.
\zt{Qwen and Mistral models have different temp settings}

Responses are parsed as JSON and matched against valid answer choices using fuzzy string matching (normalized edit similarity $\geq 0.75$) with singular/plural variation handling.
Responses that fail JSON parsing or answer matching are retried up to a configurable maximum (5--25 retries depending on model reliability).
Each experiment configuration is run as an independent grid job using our evaluation framework.

\subsection{Baseline: Fragility of Single-Agent Evaluation}\label{sec:exp-baseline}

\paragraph{Research question.}
How sensitive are fairness scores---computed under the standard single-turn paradigm---to purely cosmetic changes in the evaluation protocol?

\paragraph{Design.}
We run a full factorial sweep over the prompt-formatting parameters described in \S\ref{sec:prompt-construction}: 11 choice display formats $\times$ 2 response field orderings, yielding 22 configurations per model.
To assess sampling variance, we additionally vary temperature over $\tau \in \{0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3\}$, giving $22 \times 7 = 154$ configurations per model.
\zt{we have dropped the temp variation part, and use the suggested sampling parameter for each specific model}
For this baseline, all agents have no identity attributes (persona, demographics: \texttt{null}; \texttt{if\_as\_human}: \texttt{false}).

We first evaluate on the sampled sets from all 11 BBQ subcategories, DiscrimEval (explicit and implicit), and DifferenceAwareness across the full evaluation suite.

\paragraph{Metric.}
Following \citet{parrish2021bbq}, we compute a bias score over the disambiguated subset of BBQ questions, where a positive score indicates alignment with the stereotypical direction and a negative score indicates the anti-stereotypical direction.
For DiscrimEval we report the average decision rate disaggregated by demographic attribute \citep{tamkin2023evaluating}.
For DifferenceAwareness we report accuracy on each of the eight subsets \citep{wang2025fairness}.

\paragraph{Results.}
\st{TODO: fill in actual numbers once all formatting-variant runs complete.}

\st{TODO: Note any benchmark-specific patterns for DiscrimEval and DifferenceAwareness---e.g., whether formatting sensitivity is stronger for explicit vs.\ implicit DiscrimEval, or for factual (d1--d4) vs.\ nuanced (n1--n4) DifferenceAwareness subsets.}

\subsection{Multi-Agent Evaluation of Covert Discrimination}\label{sec:exp-multiagent}

\paragraph{Research question.} Does a model's conversational behavior---its tendency to defer, persist, or shift positions---vary systematically with the identity attributes assigned to it, even when every individual response may appear fair in isolation?

\paragraph{Identity conditions.} We test a controlled factorial combination of identity attributes for the identity-varied agent (agent $a_{\text{id}}$):
\begin{itemize}
    \item \textbf{Persona}: \{\textit{doctor}, \textit{economist}, \textit{policy expert}, $\emptyset$\}
    \item \textbf{Demographics}: \{\textit{Black}, \textit{white}, \textit{elder white female}, \textit{Hispanic}, $\emptyset$\}
    \item \textbf{Human-versus-AI framing} (\texttt{if\_as\_human}): \{\texttt{true}, \texttt{false}\}
\end{itemize}
The vanilla agent ($a_{\text{vanilla}}$) always has all attributes set to \texttt{null}/\texttt{false}.
Both agents are instantiated from the same underlying model, so any observed behavioral asymmetry is attributable solely to the assigned identity.

We evaluate under two \emph{identity reveal} conditions:
\begin{itemize}
    \item \textbf{Revealed}: the identity-varied agent's persona and demographics are visible to the vanilla agent in the conversation history (\texttt{reveal\_persona}, \texttt{reveal\_demographics}, \texttt{reveal\_presence\_mode}: all \texttt{true}).
    \item \textbf{Anonymous}: no identity information is visible to either agent (\texttt{reveal\_*}: all \texttt{false}).
\end{itemize}
Comparing these two conditions separates two mechanisms: discrimination driven by an agent's \emph{perceived} social identity (revealed condition) versus behavioral asymmetries arising from the identity's influence on the agent's own reasoning, invisible to interlocutors (anonymous condition).

Each conversation runs for $R = 3$ rounds \st{TODO: confirm final round count} over all questions in each benchmark, repeated five times per condition to control for sampling variance.
We report the behavioral metrics defined in \S\ref{sec:behavioral-metrics}: stubbornness and agreeableness rates, volatility, and their deltas ($\Delta$) between the identity-varied and vanilla agents.

\subsection{Cross-Benchmark Generality}\label{sec:exp-crossbenchmark}

\st{TODO: Once results are complete across all three benchmarks, expand this subsection to report whether the fragility of single-agent scores and the perspective asymmetry patterns generalize. Key comparisons to include:}

\begin{itemize}
    \item \textit{BBQ}: Does the pattern of $\Delta$ metrics vary systematically across the 11 demographic subcategories (e.g., is stubbornness asymmetry larger for race/ethnicity than for age or gender)?
    \item \textit{DiscrimEval}: Are asymmetries stronger under the \emph{explicit} variant (where demographics are directly stated) or the \emph{implicit} variant (where demographics are inferred from names and context), and does this interact with the identity reveal condition?
    \item \textit{DifferenceAwareness}: Since correct answers here may require acknowledging real demographic differences rather than treating groups identically, does the behavioral pattern of the identity-varied agent diverge from BBQ/DiscrimEval? Specifically, does demographic assignment affect willingness to assert factually accurate but socially charged answers (d1--d4 subsets)?
\end{itemize}

\st{TODO: Consider including a summary heatmap (models $\times$ benchmarks $\times$ identity conditions) as a figure if space permits.}

Convo snippets:
PI: shreyas and deonna, the new experimental result is pushed to HF.  ministral3-3b and qwen3-4b are done with the first experiment -- please take a look and start the analysis when you have a chance. and a note on the sample size, for bbq, it's 200 per subcategory (100 ambig, 100 disambig)
Deonna: [shares repo]

PI: several quick things upon initial go-through:
2a, in heatmap, disambig score can be negative. IMPORTANT double check the score calculation themselves.
2a, why SES is constant -1? after fixing that, the cmap bar can use the actual max instead of -1 to 1, which helps visualizing the close-to-0 values

PI: [shares doc that I paste below with rough roadmap for analysis plan]

on BBQ: Single Agent + Cosmetic Variations
Why do we need this set of exps?
to challenge the one-model-in, one-score-out LLM fairness eval via basic variations
What data do we use?
https://huggingface.co/datasets/aims-foundation/mac-fairness/tree/main/sampled-set-with-rep-exps

quick notes on the data:
for each model, we use the recommended sampling parameters, and run 1-agent 1-round on sampled questions from datasets, no persona, no demographics, as-ai (i.e., not as-human for the system prompt)
we vary the format of choice presentation (e.g., none, letter_colon, bulletin) and requested answer format (e.g., answer_first, rationale_first)

What visualization do we want?
Heatmaps for bias scores (example is for illustration of format only, ignore specific values in the cell, and model names on the left)


two heatmaps (one for disambig, one for ambig) per {choice_display_format X json_field_order} combination, essentially, a figure could be, e.g., "The heatmap illustrating the BBQ bias scores in disambiguated contexts across dataset subcategories and X models, when the choice_display_format is Y, and the json_field_order is Z" (the title of the heatmap could be sth simple and direct, like "Roman_colon X answer_first", to make it more readable)
each row is a LM, and each column is for a BBQ subcategory
each cell is averaged over the 5 rep runs, showing the mean without the +/- from reps is fine
the cmap bar can be present or hidden, when exporting the figure

What tables do we want?

Tables containing accuracies



Age
Gender
(other BBQ subcategories)
Roman numeral variants
86.2 \pm 0.6




Alphabetical variants






Arabic numeral variants






Bulletin






None
84.1 \pm 0.3






two tables (one for answer first, one for rationale first) per model (5 rows in total), OR, one table with 10 rows in total (answer first and rationale first are paired)
each entry is mean +/- std, note that for "variants", e.g., roman variants, we're combining roman_colon, roman_paren, and roman_dot to avoid table taking too much space (in appendix, it's fine to have big tables)


on BBQ: Single Agent + Identity Variations
(ZT will continue adding more details)

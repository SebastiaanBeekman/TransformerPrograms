import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys] for q in queries]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/length/rasp/hist/trainlength40/s2/hist_weights.csv",
        index_col=[0, 1],
        dtype={"feature": str},
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(token, position):
        if token in {"0", "3"}:
            return position == 16
        elif token in {"1"}:
            return position == 10
        elif token in {"2"}:
            return position == 7
        elif token in {"4"}:
            return position == 15
        elif token in {"5"}:
            return position == 26
        elif token in {"<s>"}:
            return position == 39

    attn_0_0_pattern = select_closest(positions, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 47
        elif q_position in {1, 35, 36, 38, 11}:
            return k_position == 16
        elif q_position in {26, 2, 10, 31}:
            return k_position == 14
        elif q_position in {32, 3, 4, 5, 37, 7, 8, 9, 13, 28}:
            return k_position == 17
        elif q_position in {25, 6, 15}:
            return k_position == 39
        elif q_position in {33, 12, 21, 23, 24, 27, 29}:
            return k_position == 10
        elif q_position in {39, 14, 16, 20, 22}:
            return k_position == 12
        elif q_position in {34, 17, 18, 19, 30}:
            return k_position == 9
        elif q_position in {40}:
            return k_position == 36
        elif q_position in {41}:
            return k_position == 34
        elif q_position in {42}:
            return k_position == 11
        elif q_position in {43}:
            return k_position == 30
        elif q_position in {44}:
            return k_position == 42
        elif q_position in {45}:
            return k_position == 0
        elif q_position in {46}:
            return k_position == 24
        elif q_position in {47}:
            return k_position == 38
        elif q_position in {48}:
            return k_position == 13
        elif q_position in {49}:
            return k_position == 33

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, positions)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {
            0,
            1,
            2,
            6,
            8,
            10,
            11,
            13,
            15,
            16,
            17,
            19,
            21,
            22,
            23,
            24,
            25,
            27,
            28,
            29,
            30,
            31,
            33,
            46,
            49,
        }:
            return token == "2"
        elif position in {
            3,
            4,
            5,
            7,
            9,
            12,
            14,
            18,
            20,
            26,
            32,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            44,
            45,
            47,
            48,
        }:
            return token == ""
        elif position in {41, 42, 43}:
            return token == "1"

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 45}:
            return k_position == 16
        elif q_position in {1, 2, 3, 6, 39, 9, 20, 21}:
            return k_position == 14
        elif q_position in {33, 35, 4, 11, 16, 18, 23}:
            return k_position == 8
        elif q_position in {8, 5, 7}:
            return k_position == 15
        elif q_position in {36, 10, 43, 14, 27, 31}:
            return k_position == 12
        elif q_position in {24, 34, 12, 37}:
            return k_position == 13
        elif q_position in {40, 13, 15, 17, 19, 26, 29, 30}:
            return k_position == 9
        elif q_position in {32, 38, 22, 25, 28}:
            return k_position == 10
        elif q_position in {41}:
            return k_position == 41
        elif q_position in {42}:
            return k_position == 25
        elif q_position in {44, 46}:
            return k_position == 43
        elif q_position in {47}:
            return k_position == 46
        elif q_position in {48}:
            return k_position == 11
        elif q_position in {49}:
            return k_position == 39

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, positions)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == ""

    num_attn_0_0_pattern = select(tokens, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == ""

    num_attn_0_1_pattern = select(tokens, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == ""

    num_attn_0_2_pattern = select(tokens, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == "<s>"

    num_attn_0_3_pattern = select(tokens, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_2_output):
        key = attn_0_2_output
        return 8

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in attn_0_2_outputs]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_1_output):
        key = attn_0_1_output
        return 44

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in attn_0_1_outputs]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_3_output):
        key = num_attn_0_3_output
        return 18

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_3_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_3_output):
        key = num_attn_0_3_output
        if key in {0, 1}:
            return 45
        return 42

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_3_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                token_scores,
                position_scores,
                attn_0_0_output_scores,
                attn_0_1_output_scores,
                attn_0_2_output_scores,
                attn_0_3_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
            ]
        ]
    )
    logits = feature_logits.groupby(level=0).sum(numeric_only=True).to_numpy()
    classes = classifier_weights.columns.to_numpy()
    predictions = classes[logits.argmax(-1)]
    if tokens[0] == "<s>":
        predictions[0] = "<s>"
    if tokens[-1] == "</s>":
        predictions[-1] = "</s>"
    return predictions.tolist()


print(
    run(
        [
            "<s>",
            "5",
            "0",
            "3",
            "2",
            "3",
            "0",
            "2",
            "1",
            "3",
            "5",
            "2",
            "4",
            "4",
            "4",
            "5",
            "3",
        ]
    )
)

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
        "output/length/rasp/hist/trainlength40/s1/hist_weights.csv",
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
    def predicate_0_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 22
        elif q_position in {1, 34, 37, 13, 20, 25, 27, 28, 29}:
            return k_position == 11
        elif q_position in {2, 3, 5, 6, 8, 11, 12, 17, 21, 30}:
            return k_position == 14
        elif q_position in {32, 4, 39, 9, 15, 16, 48, 18, 23, 31}:
            return k_position == 13
        elif q_position in {7}:
            return k_position == 17
        elif q_position in {33, 10, 14, 19, 22, 24}:
            return k_position == 12
        elif q_position in {26, 36, 38}:
            return k_position == 10
        elif q_position in {35}:
            return k_position == 9
        elif q_position in {40, 46}:
            return k_position == 18
        elif q_position in {41, 47}:
            return k_position == 8
        elif q_position in {42}:
            return k_position == 16
        elif q_position in {43}:
            return k_position == 30
        elif q_position in {44}:
            return k_position == 25
        elif q_position in {45}:
            return k_position == 24
        elif q_position in {49}:
            return k_position == 26

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, positions)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 40, 44, 46}:
            return k_position == 6
        elif q_position in {1, 13, 41}:
            return k_position == 36
        elif q_position in {2}:
            return k_position == 37
        elif q_position in {9, 3, 23}:
            return k_position == 33
        elif q_position in {32, 4, 37, 8, 20, 25}:
            return k_position == 38
        elif q_position in {16, 27, 5}:
            return k_position == 12
        elif q_position in {6}:
            return k_position == 16
        elif q_position in {33, 35, 7, 22, 24, 26}:
            return k_position == 13
        elif q_position in {36, 10, 11, 12, 14}:
            return k_position == 27
        elif q_position in {18, 15}:
            return k_position == 25
        elif q_position in {17}:
            return k_position == 32
        elif q_position in {19, 30, 31}:
            return k_position == 9
        elif q_position in {28, 21}:
            return k_position == 2
        elif q_position in {29}:
            return k_position == 22
        elif q_position in {34, 43, 38}:
            return k_position == 1
        elif q_position in {39}:
            return k_position == 10
        elif q_position in {42, 47}:
            return k_position == 8
        elif q_position in {48, 45}:
            return k_position == 18
        elif q_position in {49}:
            return k_position == 14

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 19
        elif q_position in {1}:
            return k_position == 18
        elif q_position in {2, 20, 36, 31}:
            return k_position == 13
        elif q_position in {3, 11, 12, 18, 26}:
            return k_position == 38
        elif q_position in {4}:
            return k_position == 41
        elif q_position in {5, 8, 9, 14, 16}:
            return k_position == 37
        elif q_position in {6, 7}:
            return k_position == 27
        elif q_position in {10}:
            return k_position == 25
        elif q_position in {19, 13, 23}:
            return k_position == 4
        elif q_position in {30, 22, 15}:
            return k_position == 36
        elif q_position in {33, 37, 17, 27, 28, 29}:
            return k_position == 10
        elif q_position in {48, 21, 38}:
            return k_position == 9
        elif q_position in {24}:
            return k_position == 2
        elif q_position in {25, 44}:
            return k_position == 15
        elif q_position in {32}:
            return k_position == 7
        elif q_position in {34, 42, 39}:
            return k_position == 24
        elif q_position in {35}:
            return k_position == 39
        elif q_position in {40}:
            return k_position == 1
        elif q_position in {41}:
            return k_position == 20
        elif q_position in {43}:
            return k_position == 8
        elif q_position in {45}:
            return k_position == 23
        elif q_position in {49, 46}:
            return k_position == 22
        elif q_position in {47}:
            return k_position == 40

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 9, 41, 43, 46}:
            return k_position == 1
        elif q_position in {8, 1}:
            return k_position == 15
        elif q_position in {33, 2, 35, 12, 13, 18, 19, 22, 31}:
            return k_position == 10
        elif q_position in {32, 3, 39, 10, 11, 14, 15, 17, 20, 27, 30}:
            return k_position == 9
        elif q_position in {4}:
            return k_position == 12
        elif q_position in {5}:
            return k_position == 47
        elif q_position in {44, 6}:
            return k_position == 37
        elif q_position in {7}:
            return k_position == 42
        elif q_position in {34, 36, 37, 38, 16, 21, 23, 24, 25, 26, 28, 29}:
            return k_position == 8
        elif q_position in {40}:
            return k_position == 31
        elif q_position in {42}:
            return k_position == 21
        elif q_position in {45}:
            return k_position == 22
        elif q_position in {47}:
            return k_position == 13
        elif q_position in {48}:
            return k_position == 17
        elif q_position in {49}:
            return k_position == 23

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
            return k_token == ""

    num_attn_0_3_pattern = select(tokens, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_3_output, attn_0_0_output):
        key = (attn_0_3_output, attn_0_0_output)
        return 28

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_0_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_3_output):
        key = attn_0_3_output
        return 28

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in attn_0_3_outputs]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output):
        key = num_attn_0_1_output
        if key in {0, 1}:
            return 10
        elif key in {2}:
            return 24
        return 28

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_1_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output):
        key = num_attn_0_1_output
        if key in {
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
        }:
            return 39
        return 14

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_1_outputs]
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
            "3",
            "4",
            "0",
            "1",
            "3",
            "5",
            "0",
            "0",
            "1",
            "4",
            "5",
            "4",
            "1",
            "2",
            "4",
            "5",
            "2",
            "4",
            "3",
            "4",
            "2",
            "4",
            "5",
            "2",
            "4",
            "1",
            "1",
            "0",
            "5",
            "1",
            "1",
            "5",
            "1",
            "1",
            "0",
            "4",
            "1",
            "0",
        ]
    )
)

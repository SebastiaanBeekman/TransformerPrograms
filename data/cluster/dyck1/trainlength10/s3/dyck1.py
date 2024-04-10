import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys[: i + 1]) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys[: i + 1]] for i, q in enumerate(queries)]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/length/rasp/dyck1/trainlength10/s3/dyck1_weights.csv",
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
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 4
        elif token in {"<s>"}:
            return position == 18

    attn_0_0_pattern = select_closest(positions, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(token, position):
        if token in {"(", ")"}:
            return position == 1
        elif token in {"<s>"}:
            return position == 7

    attn_0_1_pattern = select_closest(positions, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 1, 9, 12, 13, 14, 17}:
            return k_position == 1
        elif q_position in {8, 2}:
            return k_position == 2
        elif q_position in {16, 3}:
            return k_position == 14
        elif q_position in {4, 6, 7}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 19
        elif q_position in {18, 19, 10, 11}:
            return k_position == 4
        elif q_position in {15}:
            return k_position == 15

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 1, 10, 11, 12, 13, 14, 17, 18, 19}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {3}:
            return k_position == 8
        elif q_position in {8, 4, 7}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 9
        elif q_position in {6}:
            return k_position == 12
        elif q_position in {9}:
            return k_position == 5
        elif q_position in {15}:
            return k_position == 10
        elif q_position in {16}:
            return k_position == 19

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(token, position):
        if token in {"("}:
            return position == 5
        elif token in {")"}:
            return position == 2
        elif token in {"<s>"}:
            return position == 1

    attn_0_4_pattern = select_closest(positions, tokens, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(token, position):
        if token in {"(", "<s>"}:
            return position == 1
        elif token in {")"}:
            return position == 4

    attn_0_5_pattern = select_closest(positions, tokens, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(token, position):
        if token in {"(", ")"}:
            return position == 1
        elif token in {"<s>"}:
            return position == 11

    attn_0_6_pattern = select_closest(positions, tokens, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 2
        elif token in {"<s>"}:
            return position == 7

    attn_0_7_pattern = select_closest(positions, tokens, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"("}:
            return position == 17
        elif token in {")"}:
            return position == 0
        elif token in {"<s>"}:
            return position == 14

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 1, 2, 4, 6, 8, 11, 13, 14, 16, 17, 18, 19}:
            return token == ""
        elif position in {3, 5, 7, 10, 12, 15}:
            return token == ")"
        elif position in {9}:
            return token == "<s>"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_token, k_token):
        if q_token in {"(", "<s>"}:
            return k_token == ""
        elif q_token in {")"}:
            return k_token == "<s>"

    num_attn_0_2_pattern = select(tokens, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"("}:
            return position == 18
        elif token in {")"}:
            return position == 9
        elif token in {"<s>"}:
            return position == 3

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(q_position, k_position):
        if q_position in {0, 1, 16}:
            return k_position == 18
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 10
        elif q_position in {17, 19, 5, 14}:
            return k_position == 3
        elif q_position in {6, 7}:
            return k_position == 11
        elif q_position in {8}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 12
        elif q_position in {10}:
            return k_position == 17
        elif q_position in {11, 15}:
            return k_position == 1
        elif q_position in {12}:
            return k_position == 15
        elif q_position in {13}:
            return k_position == 14
        elif q_position in {18}:
            return k_position == 13

    num_attn_0_4_pattern = select(positions, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(q_token, k_token):
        if q_token in {"(", ")"}:
            return k_token == ""
        elif q_token in {"<s>"}:
            return k_token == ")"

    num_attn_0_5_pattern = select(tokens, tokens, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(q_position, k_position):
        if q_position in {0}:
            return k_position == 9
        elif q_position in {1, 5}:
            return k_position == 2
        elif q_position in {8, 2}:
            return k_position == 10
        elif q_position in {3, 12, 15, 7}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {19, 6}:
            return k_position == 12
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {18, 10, 11, 13}:
            return k_position == 5
        elif q_position in {14}:
            return k_position == 1
        elif q_position in {16}:
            return k_position == 4
        elif q_position in {17}:
            return k_position == 0

    num_attn_0_6_pattern = select(positions, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(q_token, k_token):
        if q_token in {"(", "<s>", ")"}:
            return k_token == ""

    num_attn_0_7_pattern = select(tokens, tokens, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output, attn_0_1_output):
        key = (attn_0_0_output, attn_0_1_output)
        return 2

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_1_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_3_output, attn_0_5_output):
        key = (attn_0_3_output, attn_0_5_output)
        if key in {(")", ")"), (")", "<s>")}:
            return 2
        elif key in {("<s>", "<s>")}:
            return 0
        return 5

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_5_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output, num_attn_0_3_output):
        key = (num_attn_0_0_output, num_attn_0_3_output)
        return 16

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_5_output, num_attn_0_7_output):
        key = (num_attn_0_5_output, num_attn_0_7_output)
        return 18

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_5_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_2_output, num_mlp_0_1_output):
        if attn_0_2_output in {"(", "<s>"}:
            return num_mlp_0_1_output == 2
        elif attn_0_2_output in {")"}:
            return num_mlp_0_1_output == 5

    attn_1_0_pattern = select_closest(
        num_mlp_0_1_outputs, attn_0_2_outputs, predicate_1_0
    )
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_2_output, attn_0_5_output):
        if attn_0_2_output in {"(", ")"}:
            return attn_0_5_output == ""
        elif attn_0_2_output in {"<s>"}:
            return attn_0_5_output == "("

    attn_1_1_pattern = select_closest(attn_0_5_outputs, attn_0_2_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_2_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_1_output, attn_0_2_output):
        if attn_0_1_output in {"(", "<s>"}:
            return attn_0_2_output == ")"
        elif attn_0_1_output in {")"}:
            return attn_0_2_output == ""

    attn_1_2_pattern = select_closest(attn_0_2_outputs, attn_0_1_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_6_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(mlp_0_1_output, attn_0_4_output):
        if mlp_0_1_output in {0, 4, 7, 9, 10, 11, 12, 13, 15, 19}:
            return attn_0_4_output == "("
        elif mlp_0_1_output in {1, 2, 6, 8, 14, 16, 17, 18}:
            return attn_0_4_output == ""
        elif mlp_0_1_output in {3}:
            return attn_0_4_output == ")"
        elif mlp_0_1_output in {5}:
            return attn_0_4_output == "<s>"

    attn_1_3_pattern = select_closest(attn_0_4_outputs, mlp_0_1_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_3_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(attn_0_1_output, position):
        if attn_0_1_output in {"(", "<s>", ")"}:
            return position == 1

    attn_1_4_pattern = select_closest(positions, attn_0_1_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_5_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(attn_0_0_output, position):
        if attn_0_0_output in {"("}:
            return position == 2
        elif attn_0_0_output in {"<s>", ")"}:
            return position == 18

    attn_1_5_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, tokens)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(token, position):
        if token in {"(", "<s>"}:
            return position == 1
        elif token in {")"}:
            return position == 2

    attn_1_6_pattern = select_closest(positions, tokens, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_6_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_6_output, position):
        if attn_0_6_output in {"(", "<s>"}:
            return position == 1
        elif attn_0_6_output in {")"}:
            return position == 2

    attn_1_7_pattern = select_closest(positions, attn_0_6_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, tokens)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(mlp_0_1_output, attn_0_2_output):
        if mlp_0_1_output in {0, 3, 4, 6, 9, 10, 12, 14, 15, 16, 19}:
            return attn_0_2_output == ")"
        elif mlp_0_1_output in {1}:
            return attn_0_2_output == "<pad>"
        elif mlp_0_1_output in {2, 5, 7, 8, 13}:
            return attn_0_2_output == ""
        elif mlp_0_1_output in {18, 11}:
            return attn_0_2_output == "("
        elif mlp_0_1_output in {17}:
            return attn_0_2_output == "<s>"

    num_attn_1_0_pattern = select(attn_0_2_outputs, mlp_0_1_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_1_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, attn_0_5_output):
        if position in {0, 4, 5, 6, 7, 10, 11, 12, 13, 15, 16, 18}:
            return attn_0_5_output == ")"
        elif position in {1, 2, 3, 8, 14, 17, 19}:
            return attn_0_5_output == ""
        elif position in {9}:
            return attn_0_5_output == "<s>"

    num_attn_1_1_pattern = select(attn_0_5_outputs, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_4_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_7_output, mlp_0_0_output):
        if attn_0_7_output in {"("}:
            return mlp_0_0_output == 14
        elif attn_0_7_output in {")"}:
            return mlp_0_0_output == 8
        elif attn_0_7_output in {"<s>"}:
            return mlp_0_0_output == 17

    num_attn_1_2_pattern = select(mlp_0_0_outputs, attn_0_7_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_7_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(token, position):
        if token in {"("}:
            return position == 2
        elif token in {")"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 5

    num_attn_1_3_pattern = select(positions, tokens, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_5_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(attn_0_6_output, attn_0_2_output):
        if attn_0_6_output in {"("}:
            return attn_0_2_output == "<s>"
        elif attn_0_6_output in {")"}:
            return attn_0_2_output == ""
        elif attn_0_6_output in {"<s>"}:
            return attn_0_2_output == ")"

    num_attn_1_4_pattern = select(attn_0_2_outputs, attn_0_6_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_0_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(attn_0_1_output, attn_0_3_output):
        if attn_0_1_output in {"(", "<s>"}:
            return attn_0_3_output == ""
        elif attn_0_1_output in {")"}:
            return attn_0_3_output == ")"

    num_attn_1_5_pattern = select(attn_0_3_outputs, attn_0_1_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_7_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(num_mlp_0_1_output, num_mlp_0_0_output):
        if num_mlp_0_1_output in {0}:
            return num_mlp_0_0_output == 0
        elif num_mlp_0_1_output in {1}:
            return num_mlp_0_0_output == 3
        elif num_mlp_0_1_output in {11, 9, 2, 10}:
            return num_mlp_0_0_output == 17
        elif num_mlp_0_1_output in {17, 3}:
            return num_mlp_0_0_output == 9
        elif num_mlp_0_1_output in {4, 12}:
            return num_mlp_0_0_output == 8
        elif num_mlp_0_1_output in {5}:
            return num_mlp_0_0_output == 16
        elif num_mlp_0_1_output in {19, 6, 15}:
            return num_mlp_0_0_output == 2
        elif num_mlp_0_1_output in {18, 14, 7}:
            return num_mlp_0_0_output == 15
        elif num_mlp_0_1_output in {8}:
            return num_mlp_0_0_output == 11
        elif num_mlp_0_1_output in {13}:
            return num_mlp_0_0_output == 6
        elif num_mlp_0_1_output in {16}:
            return num_mlp_0_0_output == 5

    num_attn_1_6_pattern = select(
        num_mlp_0_0_outputs, num_mlp_0_1_outputs, num_predicate_1_6
    )
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_7_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(attn_0_7_output, attn_0_2_output):
        if attn_0_7_output in {"("}:
            return attn_0_2_output == ")"
        elif attn_0_7_output in {"<s>", ")"}:
            return attn_0_2_output == ""

    num_attn_1_7_pattern = select(attn_0_2_outputs, attn_0_7_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_1_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_3_output, attn_1_7_output):
        key = (attn_1_3_output, attn_1_7_output)
        if key in {("(", ")"), (")", "("), (")", ")"), (")", "<s>"), ("<s>", ")")}:
            return 2
        return 17

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_1_7_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_7_output, attn_1_5_output):
        key = (attn_1_7_output, attn_1_5_output)
        return 1

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_7_outputs, attn_1_5_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_3_output, num_attn_0_1_output):
        key = (num_attn_1_3_output, num_attn_0_1_output)
        if key in {
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 10),
            (3, 3),
            (3, 4),
            (3, 5),
            (3, 6),
            (3, 7),
            (3, 8),
            (3, 9),
            (3, 10),
            (3, 11),
            (3, 12),
            (3, 13),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 7),
            (4, 8),
            (4, 9),
            (4, 10),
            (4, 11),
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 15),
            (5, 4),
            (5, 5),
            (5, 6),
            (5, 7),
            (5, 8),
            (5, 9),
            (5, 10),
            (5, 11),
            (5, 12),
            (5, 13),
            (5, 14),
            (5, 15),
            (5, 16),
            (5, 17),
            (6, 4),
            (6, 5),
            (6, 6),
            (6, 7),
            (6, 8),
            (6, 9),
            (6, 10),
            (6, 11),
            (6, 12),
            (6, 13),
            (6, 14),
            (6, 15),
            (6, 16),
            (6, 17),
            (6, 18),
            (6, 19),
            (6, 20),
            (7, 5),
            (7, 6),
            (7, 7),
            (7, 8),
            (7, 9),
            (7, 10),
            (7, 11),
            (7, 12),
            (7, 13),
            (7, 14),
            (7, 15),
            (7, 16),
            (7, 17),
            (7, 18),
            (7, 19),
            (7, 20),
            (7, 21),
            (8, 6),
            (8, 7),
            (8, 8),
            (8, 9),
            (8, 10),
            (8, 11),
            (8, 12),
            (8, 13),
            (8, 14),
            (8, 15),
            (8, 16),
            (8, 17),
            (8, 18),
            (8, 19),
            (8, 20),
            (8, 21),
            (9, 6),
            (9, 7),
            (9, 8),
            (9, 9),
            (9, 10),
            (9, 11),
            (9, 12),
            (9, 13),
            (9, 14),
            (9, 15),
            (9, 16),
            (9, 17),
            (9, 18),
            (9, 19),
            (9, 20),
            (9, 21),
            (9, 22),
            (10, 7),
            (10, 8),
            (10, 9),
            (10, 10),
            (10, 11),
            (10, 12),
            (10, 13),
            (10, 14),
            (10, 15),
            (10, 16),
            (10, 17),
            (10, 18),
            (10, 19),
            (10, 20),
            (10, 21),
            (10, 22),
            (11, 7),
            (11, 8),
            (11, 9),
            (11, 10),
            (11, 11),
            (11, 12),
            (11, 13),
            (11, 14),
            (11, 15),
            (11, 16),
            (11, 17),
            (11, 18),
            (11, 19),
            (11, 20),
            (11, 21),
            (11, 22),
            (12, 8),
            (12, 9),
            (12, 10),
            (12, 11),
            (12, 12),
            (12, 13),
            (12, 14),
            (12, 15),
            (12, 16),
            (12, 17),
            (12, 18),
            (12, 19),
            (12, 20),
            (12, 21),
            (12, 22),
            (13, 9),
            (13, 10),
            (13, 11),
            (13, 12),
            (13, 13),
            (13, 14),
            (13, 15),
            (13, 16),
            (13, 17),
            (13, 18),
            (13, 19),
            (13, 20),
            (13, 21),
            (13, 22),
            (13, 23),
            (14, 9),
            (14, 10),
            (14, 11),
            (14, 12),
            (14, 13),
            (14, 14),
            (14, 15),
            (14, 16),
            (14, 17),
            (14, 18),
            (14, 19),
            (14, 20),
            (14, 21),
            (14, 22),
            (14, 23),
            (15, 10),
            (15, 11),
            (15, 12),
            (15, 13),
            (15, 14),
            (15, 15),
            (15, 16),
            (15, 17),
            (15, 18),
            (15, 19),
            (15, 20),
            (15, 21),
            (15, 22),
            (15, 23),
            (16, 11),
            (16, 12),
            (16, 13),
            (16, 14),
            (16, 15),
            (16, 16),
            (16, 17),
            (16, 18),
            (16, 19),
            (16, 20),
            (16, 21),
            (16, 22),
            (16, 23),
            (17, 11),
            (17, 12),
            (17, 13),
            (17, 14),
            (17, 15),
            (17, 16),
            (17, 17),
            (17, 18),
            (17, 19),
            (17, 20),
            (17, 21),
            (17, 22),
            (17, 23),
            (18, 12),
            (18, 13),
            (18, 14),
            (18, 15),
            (18, 16),
            (18, 17),
            (18, 18),
            (18, 19),
            (18, 20),
            (18, 21),
            (18, 22),
            (18, 23),
            (18, 24),
            (19, 13),
            (19, 14),
            (19, 15),
            (19, 16),
            (19, 17),
            (19, 18),
            (19, 19),
            (19, 20),
            (19, 21),
            (19, 22),
            (19, 23),
            (19, 24),
            (20, 13),
            (20, 14),
            (20, 15),
            (20, 16),
            (20, 17),
            (20, 18),
            (20, 19),
            (20, 20),
            (20, 21),
            (20, 22),
            (20, 23),
            (20, 24),
            (21, 14),
            (21, 15),
            (21, 16),
            (21, 17),
            (21, 18),
            (21, 19),
            (21, 20),
            (21, 21),
            (21, 22),
            (21, 23),
            (21, 24),
            (22, 15),
            (22, 16),
            (22, 17),
            (22, 18),
            (22, 19),
            (22, 20),
            (22, 21),
            (22, 22),
            (22, 23),
            (22, 24),
            (22, 25),
            (23, 16),
            (23, 17),
            (23, 18),
            (23, 19),
            (23, 20),
            (23, 21),
            (23, 22),
            (23, 23),
            (23, 24),
            (23, 25),
            (24, 16),
            (24, 17),
            (24, 18),
            (24, 19),
            (24, 20),
            (24, 21),
            (24, 22),
            (24, 23),
            (24, 24),
            (24, 25),
            (25, 17),
            (25, 18),
            (25, 19),
            (25, 20),
            (25, 21),
            (25, 22),
            (25, 23),
            (25, 24),
            (25, 25),
            (26, 18),
            (26, 19),
            (26, 20),
            (26, 21),
            (26, 22),
            (26, 23),
            (26, 24),
            (26, 25),
            (27, 18),
            (27, 19),
            (27, 20),
            (27, 21),
            (27, 22),
            (27, 23),
            (27, 24),
            (27, 25),
            (27, 26),
            (28, 19),
            (28, 20),
            (28, 21),
            (28, 22),
            (28, 23),
            (28, 24),
            (28, 25),
            (28, 26),
            (29, 20),
            (29, 21),
            (29, 22),
            (29, 23),
            (29, 24),
            (29, 25),
            (29, 26),
            (30, 20),
            (30, 21),
            (30, 22),
            (30, 23),
            (30, 24),
            (30, 25),
            (30, 26),
            (31, 21),
            (31, 22),
            (31, 23),
            (31, 24),
            (31, 25),
            (31, 26),
            (31, 27),
            (32, 22),
            (32, 23),
            (32, 24),
            (32, 25),
            (32, 26),
            (32, 27),
            (33, 22),
            (33, 23),
            (33, 24),
            (33, 25),
            (33, 26),
            (33, 27),
            (34, 23),
            (34, 24),
            (34, 25),
            (34, 26),
            (34, 27),
            (34, 28),
            (35, 24),
            (35, 25),
            (35, 26),
            (35, 27),
            (35, 28),
            (35, 29),
            (36, 24),
            (36, 25),
            (36, 26),
            (36, 27),
            (36, 28),
            (36, 29),
            (37, 25),
            (37, 26),
            (37, 27),
            (37, 28),
            (37, 29),
            (37, 30),
            (38, 26),
            (38, 27),
            (38, 28),
            (38, 29),
            (38, 30),
            (38, 31),
            (39, 26),
            (39, 27),
            (39, 28),
            (39, 29),
            (39, 30),
            (39, 31),
            (39, 32),
        }:
            return 13
        elif key in {
            (1, 0),
            (2, 0),
            (3, 0),
            (3, 1),
            (4, 0),
            (4, 1),
            (5, 0),
            (5, 1),
            (5, 2),
            (6, 0),
            (6, 1),
            (6, 2),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (9, 4),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 5),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (12, 4),
            (12, 5),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (13, 4),
            (13, 5),
            (13, 6),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (14, 4),
            (14, 5),
            (14, 6),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (15, 4),
            (15, 5),
            (15, 6),
            (15, 7),
            (16, 0),
            (16, 1),
            (16, 2),
            (16, 3),
            (16, 4),
            (16, 5),
            (16, 6),
            (16, 7),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (17, 4),
            (17, 5),
            (17, 6),
            (17, 7),
            (17, 8),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (18, 4),
            (18, 5),
            (18, 6),
            (18, 7),
            (18, 8),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (19, 4),
            (19, 5),
            (19, 6),
            (19, 7),
            (19, 8),
            (19, 9),
            (20, 0),
            (20, 1),
            (20, 2),
            (20, 3),
            (20, 4),
            (20, 5),
            (20, 6),
            (20, 7),
            (20, 8),
            (20, 9),
            (21, 0),
            (21, 1),
            (21, 2),
            (21, 3),
            (21, 4),
            (21, 5),
            (21, 6),
            (21, 7),
            (21, 8),
            (21, 9),
            (21, 10),
            (22, 0),
            (22, 1),
            (22, 2),
            (22, 3),
            (22, 4),
            (22, 5),
            (22, 6),
            (22, 7),
            (22, 8),
            (22, 9),
            (22, 10),
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
            (23, 4),
            (23, 5),
            (23, 6),
            (23, 7),
            (23, 8),
            (23, 9),
            (23, 10),
            (23, 11),
            (24, 0),
            (24, 1),
            (24, 2),
            (24, 3),
            (24, 4),
            (24, 5),
            (24, 6),
            (24, 7),
            (24, 8),
            (24, 9),
            (24, 10),
            (24, 11),
            (25, 0),
            (25, 1),
            (25, 2),
            (25, 3),
            (25, 4),
            (25, 5),
            (25, 6),
            (25, 7),
            (25, 8),
            (25, 9),
            (25, 10),
            (25, 11),
            (25, 12),
            (26, 0),
            (26, 1),
            (26, 2),
            (26, 3),
            (26, 4),
            (26, 5),
            (26, 6),
            (26, 7),
            (26, 8),
            (26, 9),
            (26, 10),
            (26, 11),
            (26, 12),
            (27, 0),
            (27, 1),
            (27, 2),
            (27, 3),
            (27, 4),
            (27, 5),
            (27, 6),
            (27, 7),
            (27, 8),
            (27, 9),
            (27, 10),
            (27, 11),
            (27, 12),
            (27, 13),
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
            (28, 4),
            (28, 5),
            (28, 6),
            (28, 7),
            (28, 8),
            (28, 9),
            (28, 10),
            (28, 11),
            (28, 12),
            (28, 13),
            (29, 0),
            (29, 1),
            (29, 2),
            (29, 3),
            (29, 4),
            (29, 5),
            (29, 6),
            (29, 7),
            (29, 8),
            (29, 9),
            (29, 10),
            (29, 11),
            (29, 12),
            (29, 13),
            (29, 14),
            (30, 0),
            (30, 1),
            (30, 2),
            (30, 3),
            (30, 4),
            (30, 5),
            (30, 6),
            (30, 7),
            (30, 8),
            (30, 9),
            (30, 10),
            (30, 11),
            (30, 12),
            (30, 13),
            (30, 14),
            (31, 0),
            (31, 1),
            (31, 2),
            (31, 3),
            (31, 4),
            (31, 5),
            (31, 6),
            (31, 7),
            (31, 8),
            (31, 9),
            (31, 10),
            (31, 11),
            (31, 12),
            (31, 13),
            (31, 14),
            (31, 15),
            (32, 0),
            (32, 1),
            (32, 2),
            (32, 3),
            (32, 4),
            (32, 5),
            (32, 6),
            (32, 7),
            (32, 8),
            (32, 9),
            (32, 10),
            (32, 11),
            (32, 12),
            (32, 13),
            (32, 14),
            (32, 15),
            (33, 0),
            (33, 1),
            (33, 2),
            (33, 3),
            (33, 4),
            (33, 5),
            (33, 6),
            (33, 7),
            (33, 8),
            (33, 9),
            (33, 10),
            (33, 11),
            (33, 12),
            (33, 13),
            (33, 14),
            (33, 15),
            (33, 16),
            (34, 0),
            (34, 1),
            (34, 2),
            (34, 3),
            (34, 4),
            (34, 5),
            (34, 6),
            (34, 7),
            (34, 8),
            (34, 9),
            (34, 10),
            (34, 11),
            (34, 12),
            (34, 13),
            (34, 14),
            (34, 15),
            (34, 16),
            (35, 0),
            (35, 1),
            (35, 2),
            (35, 3),
            (35, 4),
            (35, 5),
            (35, 6),
            (35, 7),
            (35, 8),
            (35, 9),
            (35, 10),
            (35, 11),
            (35, 12),
            (35, 13),
            (35, 14),
            (35, 15),
            (35, 16),
            (35, 17),
            (36, 0),
            (36, 1),
            (36, 2),
            (36, 3),
            (36, 4),
            (36, 5),
            (36, 6),
            (36, 7),
            (36, 8),
            (36, 9),
            (36, 10),
            (36, 11),
            (36, 12),
            (36, 13),
            (36, 14),
            (36, 15),
            (36, 16),
            (36, 17),
            (37, 0),
            (37, 1),
            (37, 2),
            (37, 3),
            (37, 4),
            (37, 5),
            (37, 6),
            (37, 7),
            (37, 8),
            (37, 9),
            (37, 10),
            (37, 11),
            (37, 12),
            (37, 13),
            (37, 14),
            (37, 15),
            (37, 16),
            (37, 17),
            (37, 18),
            (38, 0),
            (38, 1),
            (38, 2),
            (38, 3),
            (38, 4),
            (38, 5),
            (38, 6),
            (38, 7),
            (38, 8),
            (38, 9),
            (38, 10),
            (38, 11),
            (38, 12),
            (38, 13),
            (38, 14),
            (38, 15),
            (38, 16),
            (38, 17),
            (38, 18),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
            (39, 4),
            (39, 5),
            (39, 6),
            (39, 7),
            (39, 8),
            (39, 9),
            (39, 10),
            (39, 11),
            (39, 12),
            (39, 13),
            (39, 14),
            (39, 15),
            (39, 16),
            (39, 17),
            (39, 18),
            (39, 19),
        }:
            return 12
        elif key in {
            (12, 7),
            (13, 8),
            (14, 8),
            (15, 9),
            (16, 9),
            (16, 10),
            (17, 10),
            (18, 10),
            (18, 11),
            (19, 11),
            (19, 12),
            (20, 11),
            (20, 12),
            (21, 11),
            (21, 12),
            (21, 13),
            (22, 12),
            (22, 13),
            (22, 14),
            (23, 12),
            (23, 13),
            (23, 14),
            (23, 15),
            (24, 13),
            (24, 14),
            (24, 15),
            (25, 13),
            (25, 14),
            (25, 15),
            (25, 16),
            (26, 14),
            (26, 15),
            (26, 16),
            (26, 17),
            (27, 14),
            (27, 15),
            (27, 16),
            (27, 17),
            (28, 15),
            (28, 16),
            (28, 17),
            (28, 18),
            (29, 15),
            (29, 16),
            (29, 17),
            (29, 18),
            (29, 19),
            (30, 15),
            (30, 16),
            (30, 17),
            (30, 18),
            (30, 19),
            (31, 16),
            (31, 17),
            (31, 18),
            (31, 19),
            (31, 20),
            (32, 16),
            (32, 17),
            (32, 18),
            (32, 19),
            (32, 20),
            (32, 21),
            (33, 17),
            (33, 18),
            (33, 19),
            (33, 20),
            (33, 21),
            (34, 17),
            (34, 18),
            (34, 19),
            (34, 20),
            (34, 21),
            (34, 22),
            (35, 18),
            (35, 19),
            (35, 20),
            (35, 21),
            (35, 22),
            (35, 23),
            (36, 18),
            (36, 19),
            (36, 20),
            (36, 21),
            (36, 22),
            (36, 23),
            (37, 19),
            (37, 20),
            (37, 21),
            (37, 22),
            (37, 23),
            (37, 24),
            (38, 19),
            (38, 20),
            (38, 21),
            (38, 22),
            (38, 23),
            (38, 24),
            (38, 25),
            (39, 20),
            (39, 21),
            (39, 22),
            (39, 23),
            (39, 24),
            (39, 25),
        }:
            return 16
        elif key in {
            (0, 0),
            (1, 1),
            (2, 1),
            (3, 2),
            (4, 2),
            (5, 3),
            (6, 3),
            (7, 4),
            (8, 4),
            (8, 5),
            (9, 5),
            (10, 5),
            (10, 6),
            (11, 6),
            (12, 6),
            (13, 7),
            (14, 7),
            (15, 8),
            (16, 8),
            (17, 9),
            (18, 9),
            (19, 10),
            (20, 10),
            (22, 11),
            (24, 12),
            (26, 13),
            (28, 14),
        }:
            return 9
        return 18

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_7_output):
        key = num_attn_1_7_output
        if key in {0}:
            return 8
        return 4

    num_mlp_1_1_outputs = [num_mlp_1_1(k0) for k0 in num_attn_1_7_outputs]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_1_2_output, num_mlp_0_1_output):
        if attn_1_2_output in {"("}:
            return num_mlp_0_1_output == 18
        elif attn_1_2_output in {")"}:
            return num_mlp_0_1_output == 2
        elif attn_1_2_output in {"<s>"}:
            return num_mlp_0_1_output == 1

    attn_2_0_pattern = select_closest(
        num_mlp_0_1_outputs, attn_1_2_outputs, predicate_2_0
    )
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_0_2_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(mlp_1_1_output, num_mlp_1_1_output):
        if mlp_1_1_output in {0, 18, 15, 7}:
            return num_mlp_1_1_output == 1
        elif mlp_1_1_output in {8, 1}:
            return num_mlp_1_1_output == 13
        elif mlp_1_1_output in {2}:
            return num_mlp_1_1_output == 3
        elif mlp_1_1_output in {3, 14}:
            return num_mlp_1_1_output == 19
        elif mlp_1_1_output in {4}:
            return num_mlp_1_1_output == 5
        elif mlp_1_1_output in {5}:
            return num_mlp_1_1_output == 17
        elif mlp_1_1_output in {6}:
            return num_mlp_1_1_output == 14
        elif mlp_1_1_output in {16, 9}:
            return num_mlp_1_1_output == 12
        elif mlp_1_1_output in {10}:
            return num_mlp_1_1_output == 8
        elif mlp_1_1_output in {11}:
            return num_mlp_1_1_output == 11
        elif mlp_1_1_output in {12}:
            return num_mlp_1_1_output == 2
        elif mlp_1_1_output in {19, 13}:
            return num_mlp_1_1_output == 18
        elif mlp_1_1_output in {17}:
            return num_mlp_1_1_output == 0

    attn_2_1_pattern = select_closest(
        num_mlp_1_1_outputs, mlp_1_1_outputs, predicate_2_1
    )
    attn_2_1_outputs = aggregate(attn_2_1_pattern, tokens)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_1_3_output, attn_1_6_output):
        if attn_1_3_output in {"(", "<s>"}:
            return attn_1_6_output == "<s>"
        elif attn_1_3_output in {")"}:
            return attn_1_6_output == ")"

    attn_2_2_pattern = select_closest(attn_1_6_outputs, attn_1_3_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_3_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_num_mlp_0_0_output, k_num_mlp_0_0_output):
        if q_num_mlp_0_0_output in {0, 7}:
            return k_num_mlp_0_0_output == 11
        elif q_num_mlp_0_0_output in {1, 4}:
            return k_num_mlp_0_0_output == 13
        elif q_num_mlp_0_0_output in {2, 3, 5, 8, 10, 12, 13, 18}:
            return k_num_mlp_0_0_output == 1
        elif q_num_mlp_0_0_output in {16, 11, 6}:
            return k_num_mlp_0_0_output == 10
        elif q_num_mlp_0_0_output in {9, 19}:
            return k_num_mlp_0_0_output == 2
        elif q_num_mlp_0_0_output in {14}:
            return k_num_mlp_0_0_output == 14
        elif q_num_mlp_0_0_output in {15}:
            return k_num_mlp_0_0_output == 16
        elif q_num_mlp_0_0_output in {17}:
            return k_num_mlp_0_0_output == 18

    attn_2_3_pattern = select_closest(
        num_mlp_0_0_outputs, num_mlp_0_0_outputs, predicate_2_3
    )
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_1_4_output, attn_1_2_output):
        if attn_1_4_output in {"(", "<s>"}:
            return attn_1_2_output == "("
        elif attn_1_4_output in {")"}:
            return attn_1_2_output == "<s>"

    attn_2_4_pattern = select_closest(attn_1_2_outputs, attn_1_4_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_0_0_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_1_1_output, position):
        if attn_1_1_output in {"("}:
            return position == 10
        elif attn_1_1_output in {"<s>", ")"}:
            return position == 2

    attn_2_5_pattern = select_closest(positions, attn_1_1_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_0_7_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_1_6_output, attn_0_7_output):
        if attn_1_6_output in {"("}:
            return attn_0_7_output == ")"
        elif attn_1_6_output in {")"}:
            return attn_0_7_output == ""
        elif attn_1_6_output in {"<s>"}:
            return attn_0_7_output == "("

    attn_2_6_pattern = select_closest(attn_0_7_outputs, attn_1_6_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_0_7_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_1_5_output, attn_0_4_output):
        if attn_1_5_output in {"(", ")"}:
            return attn_0_4_output == ""
        elif attn_1_5_output in {"<s>"}:
            return attn_0_4_output == "<s>"

    attn_2_7_pattern = select_closest(attn_0_4_outputs, attn_1_5_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_0_7_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_0_3_output, attn_1_5_output):
        if attn_0_3_output in {"("}:
            return attn_1_5_output == ")"
        elif attn_0_3_output in {"<s>", ")"}:
            return attn_1_5_output == ""

    num_attn_2_0_pattern = select(attn_1_5_outputs, attn_0_3_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_4_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_4_output, attn_0_2_output):
        if attn_1_4_output in {"("}:
            return attn_0_2_output == ")"
        elif attn_1_4_output in {"<s>", ")"}:
            return attn_0_2_output == ""

    num_attn_2_1_pattern = select(attn_0_2_outputs, attn_1_4_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_0_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(token, attn_1_7_output):
        if token in {"("}:
            return attn_1_7_output == ")"
        elif token in {"<s>", ")"}:
            return attn_1_7_output == ""

    num_attn_2_2_pattern = select(attn_1_7_outputs, tokens, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(q_num_mlp_1_0_output, k_num_mlp_1_0_output):
        if q_num_mlp_1_0_output in {0, 7}:
            return k_num_mlp_1_0_output == 7
        elif q_num_mlp_1_0_output in {1, 9}:
            return k_num_mlp_1_0_output == 3
        elif q_num_mlp_1_0_output in {2, 3}:
            return k_num_mlp_1_0_output == 10
        elif q_num_mlp_1_0_output in {17, 4}:
            return k_num_mlp_1_0_output == 4
        elif q_num_mlp_1_0_output in {18, 5, 15}:
            return k_num_mlp_1_0_output == 1
        elif q_num_mlp_1_0_output in {16, 13, 6}:
            return k_num_mlp_1_0_output == 17
        elif q_num_mlp_1_0_output in {8, 19, 11}:
            return k_num_mlp_1_0_output == 9
        elif q_num_mlp_1_0_output in {10}:
            return k_num_mlp_1_0_output == 5
        elif q_num_mlp_1_0_output in {12}:
            return k_num_mlp_1_0_output == 15
        elif q_num_mlp_1_0_output in {14}:
            return k_num_mlp_1_0_output == 18

    num_attn_2_3_pattern = select(
        num_mlp_1_0_outputs, num_mlp_1_0_outputs, num_predicate_2_3
    )
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_5_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_1_0_output, attn_1_2_output):
        if attn_1_0_output in {0, 1, 7}:
            return attn_1_2_output == "("
        elif attn_1_0_output in {
            2,
            3,
            4,
            5,
            6,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
        }:
            return attn_1_2_output == ""

    num_attn_2_4_pattern = select(attn_1_2_outputs, attn_1_0_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_2_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(num_mlp_1_1_output, mlp_0_0_output):
        if num_mlp_1_1_output in {0, 10, 12}:
            return mlp_0_0_output == 12
        elif num_mlp_1_1_output in {1, 18, 4}:
            return mlp_0_0_output == 17
        elif num_mlp_1_1_output in {2}:
            return mlp_0_0_output == 9
        elif num_mlp_1_1_output in {16, 3}:
            return mlp_0_0_output == 18
        elif num_mlp_1_1_output in {5}:
            return mlp_0_0_output == 3
        elif num_mlp_1_1_output in {13, 6}:
            return mlp_0_0_output == 7
        elif num_mlp_1_1_output in {7}:
            return mlp_0_0_output == 14
        elif num_mlp_1_1_output in {8, 9}:
            return mlp_0_0_output == 4
        elif num_mlp_1_1_output in {11}:
            return mlp_0_0_output == 5
        elif num_mlp_1_1_output in {14}:
            return mlp_0_0_output == 10
        elif num_mlp_1_1_output in {15}:
            return mlp_0_0_output == 0
        elif num_mlp_1_1_output in {17}:
            return mlp_0_0_output == 11
        elif num_mlp_1_1_output in {19}:
            return mlp_0_0_output == 15

    num_attn_2_5_pattern = select(
        mlp_0_0_outputs, num_mlp_1_1_outputs, num_predicate_2_5
    )
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_1_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_6_output, attn_0_2_output):
        if attn_1_6_output in {"("}:
            return attn_0_2_output == ""
        elif attn_1_6_output in {"<s>", ")"}:
            return attn_0_2_output == ")"

    num_attn_2_6_pattern = select(attn_0_2_outputs, attn_1_6_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_1_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_0_6_output, attn_0_2_output):
        if attn_0_6_output in {"("}:
            return attn_0_2_output == "("
        elif attn_0_6_output in {"<s>", ")"}:
            return attn_0_2_output == ""

    num_attn_2_7_pattern = select(attn_0_2_outputs, attn_0_6_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_1_5_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(position, attn_2_0_output):
        key = (position, attn_2_0_output)
        if key in {
            (0, "<s>"),
            (4, "<s>"),
            (6, "("),
            (6, "<s>"),
            (7, "("),
            (7, "<s>"),
            (9, "("),
            (11, "<s>"),
            (12, "<s>"),
        }:
            return 15
        elif key in {(5, "("), (5, "<s>"), (17, "<s>")}:
            return 10
        elif key in {(1, "<s>"), (9, "<s>")}:
            return 7
        return 13

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(positions, attn_2_0_outputs)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_1_0_output, attn_1_7_output):
        key = (attn_1_0_output, attn_1_7_output)
        if key in {
            (0, ")"),
            (1, ")"),
            (2, ")"),
            (3, ")"),
            (4, ")"),
            (5, ")"),
            (6, ")"),
            (7, ")"),
            (8, ")"),
            (9, ")"),
            (10, ")"),
            (11, ")"),
            (12, ")"),
            (13, ")"),
            (14, ")"),
            (15, ")"),
            (16, ")"),
            (17, ")"),
            (18, ")"),
            (19, ")"),
        }:
            return 13
        return 0

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_1_0_outputs, attn_1_7_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_1_output, num_attn_2_3_output):
        key = (num_attn_1_1_output, num_attn_2_3_output)
        if key in {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
            (2, 2),
            (3, 0),
            (3, 1),
            (3, 2),
            (4, 0),
            (4, 1),
            (4, 2),
            (5, 0),
            (5, 1),
            (5, 2),
            (6, 0),
            (6, 1),
            (6, 2),
            (7, 0),
            (7, 1),
            (7, 2),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (13, 4),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (14, 4),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (15, 4),
            (16, 0),
            (16, 1),
            (16, 2),
            (16, 3),
            (16, 4),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (17, 4),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (18, 4),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (19, 4),
            (19, 5),
            (20, 0),
            (20, 1),
            (20, 2),
            (20, 3),
            (20, 4),
            (20, 5),
            (21, 0),
            (21, 1),
            (21, 2),
            (21, 3),
            (21, 4),
            (21, 5),
            (22, 0),
            (22, 1),
            (22, 2),
            (22, 3),
            (22, 4),
            (22, 5),
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
            (23, 4),
            (23, 5),
            (24, 0),
            (24, 1),
            (24, 2),
            (24, 3),
            (24, 4),
            (24, 5),
            (24, 6),
            (25, 0),
            (25, 1),
            (25, 2),
            (25, 3),
            (25, 4),
            (25, 5),
            (25, 6),
            (26, 0),
            (26, 1),
            (26, 2),
            (26, 3),
            (26, 4),
            (26, 5),
            (26, 6),
            (27, 0),
            (27, 1),
            (27, 2),
            (27, 3),
            (27, 4),
            (27, 5),
            (27, 6),
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
            (28, 4),
            (28, 5),
            (28, 6),
            (29, 0),
            (29, 1),
            (29, 2),
            (29, 3),
            (29, 4),
            (29, 5),
            (29, 6),
            (30, 0),
            (30, 1),
            (30, 2),
            (30, 3),
            (30, 4),
            (30, 5),
            (30, 6),
            (30, 7),
            (31, 0),
            (31, 1),
            (31, 2),
            (31, 3),
            (31, 4),
            (31, 5),
            (31, 6),
            (31, 7),
            (32, 0),
            (32, 1),
            (32, 2),
            (32, 3),
            (32, 4),
            (32, 5),
            (32, 6),
            (32, 7),
            (33, 0),
            (33, 1),
            (33, 2),
            (33, 3),
            (33, 4),
            (33, 5),
            (33, 6),
            (33, 7),
            (34, 0),
            (34, 1),
            (34, 2),
            (34, 3),
            (34, 4),
            (34, 5),
            (34, 6),
            (34, 7),
            (35, 0),
            (35, 1),
            (35, 2),
            (35, 3),
            (35, 4),
            (35, 5),
            (35, 6),
            (35, 7),
            (35, 8),
            (36, 0),
            (36, 1),
            (36, 2),
            (36, 3),
            (36, 4),
            (36, 5),
            (36, 6),
            (36, 7),
            (36, 8),
            (37, 0),
            (37, 1),
            (37, 2),
            (37, 3),
            (37, 4),
            (37, 5),
            (37, 6),
            (37, 7),
            (37, 8),
            (38, 0),
            (38, 1),
            (38, 2),
            (38, 3),
            (38, 4),
            (38, 5),
            (38, 6),
            (38, 7),
            (38, 8),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
            (39, 4),
            (39, 5),
            (39, 6),
            (39, 7),
            (39, 8),
            (40, 0),
            (40, 1),
            (40, 2),
            (40, 3),
            (40, 4),
            (40, 5),
            (40, 6),
            (40, 7),
            (40, 8),
            (40, 9),
            (41, 0),
            (41, 1),
            (41, 2),
            (41, 3),
            (41, 4),
            (41, 5),
            (41, 6),
            (41, 7),
            (41, 8),
            (41, 9),
            (42, 0),
            (42, 1),
            (42, 2),
            (42, 3),
            (42, 4),
            (42, 5),
            (42, 6),
            (42, 7),
            (42, 8),
            (42, 9),
            (43, 0),
            (43, 1),
            (43, 2),
            (43, 3),
            (43, 4),
            (43, 5),
            (43, 6),
            (43, 7),
            (43, 8),
            (43, 9),
            (44, 0),
            (44, 1),
            (44, 2),
            (44, 3),
            (44, 4),
            (44, 5),
            (44, 6),
            (44, 7),
            (44, 8),
            (44, 9),
            (45, 0),
            (45, 1),
            (45, 2),
            (45, 3),
            (45, 4),
            (45, 5),
            (45, 6),
            (45, 7),
            (45, 8),
            (45, 9),
            (46, 0),
            (46, 1),
            (46, 2),
            (46, 3),
            (46, 4),
            (46, 5),
            (46, 6),
            (46, 7),
            (46, 8),
            (46, 9),
            (46, 10),
            (47, 0),
            (47, 1),
            (47, 2),
            (47, 3),
            (47, 4),
            (47, 5),
            (47, 6),
            (47, 7),
            (47, 8),
            (47, 9),
            (47, 10),
            (48, 0),
            (48, 1),
            (48, 2),
            (48, 3),
            (48, 4),
            (48, 5),
            (48, 6),
            (48, 7),
            (48, 8),
            (48, 9),
            (48, 10),
            (49, 0),
            (49, 1),
            (49, 2),
            (49, 3),
            (49, 4),
            (49, 5),
            (49, 6),
            (49, 7),
            (49, 8),
            (49, 9),
            (49, 10),
            (50, 0),
            (50, 1),
            (50, 2),
            (50, 3),
            (50, 4),
            (50, 5),
            (50, 6),
            (50, 7),
            (50, 8),
            (50, 9),
            (50, 10),
            (51, 0),
            (51, 1),
            (51, 2),
            (51, 3),
            (51, 4),
            (51, 5),
            (51, 6),
            (51, 7),
            (51, 8),
            (51, 9),
            (51, 10),
            (51, 11),
            (52, 0),
            (52, 1),
            (52, 2),
            (52, 3),
            (52, 4),
            (52, 5),
            (52, 6),
            (52, 7),
            (52, 8),
            (52, 9),
            (52, 10),
            (52, 11),
            (53, 0),
            (53, 1),
            (53, 2),
            (53, 3),
            (53, 4),
            (53, 5),
            (53, 6),
            (53, 7),
            (53, 8),
            (53, 9),
            (53, 10),
            (53, 11),
            (54, 0),
            (54, 1),
            (54, 2),
            (54, 3),
            (54, 4),
            (54, 5),
            (54, 6),
            (54, 7),
            (54, 8),
            (54, 9),
            (54, 10),
            (54, 11),
            (55, 0),
            (55, 1),
            (55, 2),
            (55, 3),
            (55, 4),
            (55, 5),
            (55, 6),
            (55, 7),
            (55, 8),
            (55, 9),
            (55, 10),
            (55, 11),
            (56, 0),
            (56, 1),
            (56, 2),
            (56, 3),
            (56, 4),
            (56, 5),
            (56, 6),
            (56, 7),
            (56, 8),
            (56, 9),
            (56, 10),
            (56, 11),
            (57, 0),
            (57, 1),
            (57, 2),
            (57, 3),
            (57, 4),
            (57, 5),
            (57, 6),
            (57, 7),
            (57, 8),
            (57, 9),
            (57, 10),
            (57, 11),
            (57, 12),
            (58, 0),
            (58, 1),
            (58, 2),
            (58, 3),
            (58, 4),
            (58, 5),
            (58, 6),
            (58, 7),
            (58, 8),
            (58, 9),
            (58, 10),
            (58, 11),
            (58, 12),
            (59, 0),
            (59, 1),
            (59, 2),
            (59, 3),
            (59, 4),
            (59, 5),
            (59, 6),
            (59, 7),
            (59, 8),
            (59, 9),
            (59, 10),
            (59, 11),
            (59, 12),
        }:
            return 7
        elif key in {
            (0, 2),
            (1, 2),
            (5, 3),
            (6, 3),
            (7, 3),
            (11, 4),
            (12, 4),
            (16, 5),
            (17, 5),
            (18, 5),
            (22, 6),
            (23, 6),
            (27, 7),
            (28, 7),
            (29, 7),
            (33, 8),
            (34, 8),
            (38, 9),
            (39, 9),
            (43, 10),
            (44, 10),
            (45, 10),
            (49, 11),
            (50, 11),
            (54, 12),
            (55, 12),
            (56, 12),
        }:
            return 5
        return 2

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_2_3_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_2_output, num_attn_2_0_output):
        key = (num_attn_1_2_output, num_attn_2_0_output)
        if key in {(0, 0), (0, 1), (0, 2), (1, 0)}:
            return 5
        return 9

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_2_0_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
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
                attn_0_4_output_scores,
                attn_0_5_output_scores,
                attn_0_6_output_scores,
                attn_0_7_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                attn_1_4_output_scores,
                attn_1_5_output_scores,
                attn_1_6_output_scores,
                attn_1_7_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                attn_2_4_output_scores,
                attn_2_5_output_scores,
                attn_2_6_output_scores,
                attn_2_7_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_0_4_output_scores,
                num_attn_0_5_output_scores,
                num_attn_0_6_output_scores,
                num_attn_0_7_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_1_4_output_scores,
                num_attn_1_5_output_scores,
                num_attn_1_6_output_scores,
                num_attn_1_7_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
                num_attn_2_4_output_scores,
                num_attn_2_5_output_scores,
                num_attn_2_6_output_scores,
                num_attn_2_7_output_scores,
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


print(run(["<s>", "(", ")", ")", "(", "(", "(", ")", ")", ")"]))

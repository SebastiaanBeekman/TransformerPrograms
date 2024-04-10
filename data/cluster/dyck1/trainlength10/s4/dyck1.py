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
        "output/length/rasp/dyck1/trainlength10/s4/dyck1_weights.csv",
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
            return k_position == 17
        elif q_position in {1, 5}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {3}:
            return k_position == 14
        elif q_position in {4}:
            return k_position == 13
        elif q_position in {11, 6}:
            return k_position == 4
        elif q_position in {9, 17, 14, 7}:
            return k_position == 0
        elif q_position in {8, 18, 12, 15}:
            return k_position == 6
        elif q_position in {10}:
            return k_position == 3
        elif q_position in {13}:
            return k_position == 7
        elif q_position in {16}:
            return k_position == 18
        elif q_position in {19}:
            return k_position == 16

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 2
        elif token in {"<s>"}:
            return position == 18

    attn_0_1_pattern = select_closest(positions, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 18
        elif q_position in {1, 2, 10, 11, 12, 13, 14, 16, 17}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 12
        elif q_position in {8, 19, 4, 7}:
            return k_position == 3
        elif q_position in {5}:
            return k_position == 11
        elif q_position in {9, 6}:
            return k_position == 5
        elif q_position in {15}:
            return k_position == 10
        elif q_position in {18}:
            return k_position == 19

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 7
        elif q_position in {1, 2, 13, 15}:
            return k_position == 1
        elif q_position in {3, 4, 5, 6, 7, 9}:
            return k_position == 3
        elif q_position in {8}:
            return k_position == 13
        elif q_position in {10, 18}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 16
        elif q_position in {12}:
            return k_position == 0
        elif q_position in {17, 14}:
            return k_position == 8
        elif q_position in {16}:
            return k_position == 4
        elif q_position in {19}:
            return k_position == 19

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 2
        elif token in {"<s>"}:
            return position == 7

    attn_0_4_pattern = select_closest(positions, tokens, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(token, position):
        if token in {"("}:
            return position == 1
        elif token in {"<s>", ")"}:
            return position == 5

    attn_0_5_pattern = select_closest(positions, tokens, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 2}:
            return k_position == 11
        elif q_position in {1, 3, 5, 10, 13, 14, 15, 18, 19}:
            return k_position == 1
        elif q_position in {8, 4, 6, 7}:
            return k_position == 3
        elif q_position in {9}:
            return k_position == 5
        elif q_position in {11}:
            return k_position == 18
        elif q_position in {12}:
            return k_position == 10
        elif q_position in {16}:
            return k_position == 8
        elif q_position in {17}:
            return k_position == 17

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0, 10}:
            return k_position == 9
        elif q_position in {1, 3, 7, 9, 13, 16, 17, 18, 19}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 16
        elif q_position in {4}:
            return k_position == 8
        elif q_position in {5}:
            return k_position == 2
        elif q_position in {8, 6, 14}:
            return k_position == 5
        elif q_position in {11}:
            return k_position == 18
        elif q_position in {12}:
            return k_position == 10
        elif q_position in {15}:
            return k_position == 19

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"("}:
            return position == 4
        elif token in {"<s>", ")"}:
            return position == 2

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {"("}:
            return position == 0
        elif token in {")"}:
            return position == 15
        elif token in {"<s>"}:
            return position == 1

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0, 16, 19}:
            return k_position == 16
        elif q_position in {1, 9}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 15
        elif q_position in {5, 7}:
            return k_position == 1
        elif q_position in {10, 13, 6}:
            return k_position == 14
        elif q_position in {8, 12}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 11
        elif q_position in {14}:
            return k_position == 17
        elif q_position in {15}:
            return k_position == 12
        elif q_position in {17}:
            return k_position == 13
        elif q_position in {18}:
            return k_position == 18

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"("}:
            return position == 0
        elif token in {")"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 1

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19}:
            return token == ""
        elif position in {1}:
            return token == "<pad>"
        elif position in {3, 13}:
            return token == "("

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(q_token, k_token):
        if q_token in {"("}:
            return k_token == "<pad>"
        elif q_token in {"<s>", ")"}:
            return k_token == ""

    num_attn_0_5_pattern = select(tokens, tokens, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(token, position):
        if token in {"<s>", "("}:
            return position == 2
        elif token in {")"}:
            return position == 11

    num_attn_0_6_pattern = select(positions, tokens, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(token, position):
        if token in {"<s>", "("}:
            return position == 6
        elif token in {")"}:
            return position == 16

    num_attn_0_7_pattern = select(positions, tokens, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_5_output):
        key = attn_0_5_output
        if key in {"", "("}:
            return 1
        return 17

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in attn_0_5_outputs]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(token, attn_0_5_output):
        key = (token, attn_0_5_output)
        if key in {("(", ")"), (")", ")"), (")", "<s>"), ("<s>", ")")}:
            return 17
        return 0

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(tokens, attn_0_5_outputs)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output, num_attn_0_2_output):
        key = (num_attn_0_1_output, num_attn_0_2_output)
        return 14

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_0_output, num_attn_0_1_output):
        key = (num_attn_0_0_output, num_attn_0_1_output)
        return 15

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(num_mlp_0_0_output, position):
        if num_mlp_0_0_output in {0}:
            return position == 5
        elif num_mlp_0_0_output in {1, 14}:
            return position == 16
        elif num_mlp_0_0_output in {2}:
            return position == 0
        elif num_mlp_0_0_output in {3}:
            return position == 18
        elif num_mlp_0_0_output in {4, 15}:
            return position == 2
        elif num_mlp_0_0_output in {5, 7, 9, 10, 11, 12, 16, 17, 18, 19}:
            return position == 1
        elif num_mlp_0_0_output in {6}:
            return position == 4
        elif num_mlp_0_0_output in {8}:
            return position == 8
        elif num_mlp_0_0_output in {13}:
            return position == 13

    attn_1_0_pattern = select_closest(positions, num_mlp_0_0_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_2_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_1_output, num_mlp_0_1_output):
        if attn_0_1_output in {"("}:
            return num_mlp_0_1_output == 2
        elif attn_0_1_output in {")"}:
            return num_mlp_0_1_output == 14
        elif attn_0_1_output in {"<s>"}:
            return num_mlp_0_1_output == 11

    attn_1_1_pattern = select_closest(
        num_mlp_0_1_outputs, attn_0_1_outputs, predicate_1_1
    )
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_5_output, attn_0_0_output):
        if attn_0_5_output in {"("}:
            return attn_0_0_output == ""
        elif attn_0_5_output in {"<s>", ")"}:
            return attn_0_0_output == "("

    attn_1_2_pattern = select_closest(attn_0_0_outputs, attn_0_5_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_5_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_3_output, attn_0_4_output):
        if attn_0_3_output in {"("}:
            return attn_0_4_output == ")"
        elif attn_0_3_output in {"<s>", ")"}:
            return attn_0_4_output == ""

    attn_1_3_pattern = select_closest(attn_0_4_outputs, attn_0_3_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_7_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(attn_0_5_output, position):
        if attn_0_5_output in {"("}:
            return position == 2
        elif attn_0_5_output in {"<s>", ")"}:
            return position == 1

    attn_1_4_pattern = select_closest(positions, attn_0_5_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, tokens)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(attn_0_0_output, position):
        if attn_0_0_output in {"<s>", "("}:
            return position == 1
        elif attn_0_0_output in {")"}:
            return position == 17

    attn_1_5_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_5_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(attn_0_5_output, num_mlp_0_1_output):
        if attn_0_5_output in {"("}:
            return num_mlp_0_1_output == 2
        elif attn_0_5_output in {")"}:
            return num_mlp_0_1_output == 5
        elif attn_0_5_output in {"<s>"}:
            return num_mlp_0_1_output == 19

    attn_1_6_pattern = select_closest(
        num_mlp_0_1_outputs, attn_0_5_outputs, predicate_1_6
    )
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_5_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(q_num_mlp_0_0_output, k_num_mlp_0_0_output):
        if q_num_mlp_0_0_output in {0, 6, 10, 16, 17}:
            return k_num_mlp_0_0_output == 2
        elif q_num_mlp_0_0_output in {1}:
            return k_num_mlp_0_0_output == 13
        elif q_num_mlp_0_0_output in {2, 3, 11, 15, 18, 19}:
            return k_num_mlp_0_0_output == 1
        elif q_num_mlp_0_0_output in {4}:
            return k_num_mlp_0_0_output == 10
        elif q_num_mlp_0_0_output in {5}:
            return k_num_mlp_0_0_output == 11
        elif q_num_mlp_0_0_output in {7}:
            return k_num_mlp_0_0_output == 16
        elif q_num_mlp_0_0_output in {8, 14}:
            return k_num_mlp_0_0_output == 14
        elif q_num_mlp_0_0_output in {9, 13}:
            return k_num_mlp_0_0_output == 3
        elif q_num_mlp_0_0_output in {12}:
            return k_num_mlp_0_0_output == 15

    attn_1_7_pattern = select_closest(
        num_mlp_0_0_outputs, num_mlp_0_0_outputs, predicate_1_7
    )
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_5_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, mlp_0_1_output):
        if position in {0, 1, 19, 16}:
            return mlp_0_1_output == 4
        elif position in {2}:
            return mlp_0_1_output == 7
        elif position in {3}:
            return mlp_0_1_output == 12
        elif position in {4, 6}:
            return mlp_0_1_output == 6
        elif position in {5}:
            return mlp_0_1_output == 19
        elif position in {7}:
            return mlp_0_1_output == 10
        elif position in {8}:
            return mlp_0_1_output == 3
        elif position in {9, 13, 15}:
            return mlp_0_1_output == 17
        elif position in {10, 14}:
            return mlp_0_1_output == 15
        elif position in {11, 12}:
            return mlp_0_1_output == 8
        elif position in {17, 18}:
            return mlp_0_1_output == 13

    num_attn_1_0_pattern = select(mlp_0_1_outputs, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_6_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_3_output, attn_0_2_output):
        if attn_0_3_output in {"<s>", ")", "("}:
            return attn_0_2_output == ""

    num_attn_1_1_pattern = select(attn_0_2_outputs, attn_0_3_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_3_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(token, num_mlp_0_1_output):
        if token in {"<s>", "("}:
            return num_mlp_0_1_output == 9
        elif token in {")"}:
            return num_mlp_0_1_output == 7

    num_attn_1_2_pattern = select(num_mlp_0_1_outputs, tokens, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_1_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_0_output, num_mlp_0_1_output):
        if attn_0_0_output in {"("}:
            return num_mlp_0_1_output == 0
        elif attn_0_0_output in {")"}:
            return num_mlp_0_1_output == 19
        elif attn_0_0_output in {"<s>"}:
            return num_mlp_0_1_output == 1

    num_attn_1_3_pattern = select(
        num_mlp_0_1_outputs, attn_0_0_outputs, num_predicate_1_3
    )
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_7_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(num_mlp_0_0_output, num_mlp_0_1_output):
        if num_mlp_0_0_output in {0, 12}:
            return num_mlp_0_1_output == 19
        elif num_mlp_0_0_output in {1}:
            return num_mlp_0_1_output == 0
        elif num_mlp_0_0_output in {2}:
            return num_mlp_0_1_output == 9
        elif num_mlp_0_0_output in {3}:
            return num_mlp_0_1_output == 8
        elif num_mlp_0_0_output in {4}:
            return num_mlp_0_1_output == 7
        elif num_mlp_0_0_output in {16, 9, 5}:
            return num_mlp_0_1_output == 1
        elif num_mlp_0_0_output in {6, 7}:
            return num_mlp_0_1_output == 5
        elif num_mlp_0_0_output in {8, 10}:
            return num_mlp_0_1_output == 18
        elif num_mlp_0_0_output in {11}:
            return num_mlp_0_1_output == 12
        elif num_mlp_0_0_output in {13}:
            return num_mlp_0_1_output == 4
        elif num_mlp_0_0_output in {14}:
            return num_mlp_0_1_output == 10
        elif num_mlp_0_0_output in {15}:
            return num_mlp_0_1_output == 15
        elif num_mlp_0_0_output in {17, 18, 19}:
            return num_mlp_0_1_output == 3

    num_attn_1_4_pattern = select(
        num_mlp_0_1_outputs, num_mlp_0_0_outputs, num_predicate_1_4
    )
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_2_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 17
        elif token in {"<s>"}:
            return position == 6

    num_attn_1_5_pattern = select(positions, tokens, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_3_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(mlp_0_1_output, num_mlp_0_1_output):
        if mlp_0_1_output in {0, 4, 12}:
            return num_mlp_0_1_output == 16
        elif mlp_0_1_output in {1}:
            return num_mlp_0_1_output == 0
        elif mlp_0_1_output in {2}:
            return num_mlp_0_1_output == 13
        elif mlp_0_1_output in {3, 13}:
            return num_mlp_0_1_output == 5
        elif mlp_0_1_output in {17, 18, 5, 7}:
            return num_mlp_0_1_output == 10
        elif mlp_0_1_output in {8, 6}:
            return num_mlp_0_1_output == 15
        elif mlp_0_1_output in {9, 10}:
            return num_mlp_0_1_output == 4
        elif mlp_0_1_output in {19, 11}:
            return num_mlp_0_1_output == 1
        elif mlp_0_1_output in {14}:
            return num_mlp_0_1_output == 8
        elif mlp_0_1_output in {15}:
            return num_mlp_0_1_output == 14
        elif mlp_0_1_output in {16}:
            return num_mlp_0_1_output == 17

    num_attn_1_6_pattern = select(
        num_mlp_0_1_outputs, mlp_0_1_outputs, num_predicate_1_6
    )
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_5_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(attn_0_6_output, position):
        if attn_0_6_output in {"("}:
            return position == 2
        elif attn_0_6_output in {")"}:
            return position == 16
        elif attn_0_6_output in {"<s>"}:
            return position == 17

    num_attn_1_7_pattern = select(positions, attn_0_6_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_4_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_1_output, attn_0_2_output):
        key = (attn_1_1_output, attn_0_2_output)
        if key in {("(", ")"), (")", ")"), ("<s>", ")")}:
            return 10
        return 0

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_0_2_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_2_output, attn_1_1_output):
        key = (attn_1_2_output, attn_1_1_output)
        return 4

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_1_1_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_1_output, num_attn_1_6_output):
        key = (num_attn_1_1_output, num_attn_1_6_output)
        if key in {
            (0, 34),
            (0, 35),
            (0, 36),
            (0, 37),
            (0, 38),
            (0, 39),
            (32, 1),
            (33, 1),
            (34, 1),
            (35, 1),
            (36, 1),
            (37, 1),
            (38, 1),
            (39, 1),
        }:
            return 4
        return 15

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_6_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_1_output, num_attn_1_6_output):
        key = (num_attn_1_1_output, num_attn_1_6_output)
        return 6

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_6_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(num_mlp_0_1_output, position):
        if num_mlp_0_1_output in {0, 1, 6, 7, 9, 10, 13, 14, 17, 18}:
            return position == 1
        elif num_mlp_0_1_output in {2, 3, 4}:
            return position == 15
        elif num_mlp_0_1_output in {5, 8, 12, 15, 16}:
            return position == 2
        elif num_mlp_0_1_output in {11}:
            return position == 16
        elif num_mlp_0_1_output in {19}:
            return position == 11

    attn_2_0_pattern = select_closest(positions, num_mlp_0_1_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, tokens)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(mlp_0_0_output, position):
        if mlp_0_0_output in {0, 8, 9, 11, 14, 15, 16, 19}:
            return position == 1
        elif mlp_0_0_output in {1}:
            return position == 18
        elif mlp_0_0_output in {2, 4, 5, 6, 7, 12, 18}:
            return position == 2
        elif mlp_0_0_output in {3}:
            return position == 10
        elif mlp_0_0_output in {10}:
            return position == 13
        elif mlp_0_0_output in {13}:
            return position == 19
        elif mlp_0_0_output in {17}:
            return position == 9

    attn_2_1_pattern = select_closest(positions, mlp_0_0_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_0_5_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_1_1_output, attn_0_4_output):
        if attn_1_1_output in {")", "("}:
            return attn_0_4_output == ")"
        elif attn_1_1_output in {"<s>"}:
            return attn_0_4_output == ""

    attn_2_2_pattern = select_closest(attn_0_4_outputs, attn_1_1_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_0_7_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_1_output, attn_0_3_output):
        if attn_0_1_output in {"<s>", "("}:
            return attn_0_3_output == ""
        elif attn_0_1_output in {")"}:
            return attn_0_3_output == ")"

    attn_2_3_pattern = select_closest(attn_0_3_outputs, attn_0_1_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_5_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(mlp_0_0_output, attn_0_3_output):
        if mlp_0_0_output in {0, 3, 4, 6, 10, 11, 14, 15, 17, 19}:
            return attn_0_3_output == "("
        elif mlp_0_0_output in {1, 2}:
            return attn_0_3_output == ")"
        elif mlp_0_0_output in {5, 7, 8, 12, 13, 16, 18}:
            return attn_0_3_output == ""
        elif mlp_0_0_output in {9}:
            return attn_0_3_output == "<s>"

    attn_2_4_pattern = select_closest(attn_0_3_outputs, mlp_0_0_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_0_1_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(q_attn_0_7_output, k_attn_0_7_output):
        if q_attn_0_7_output in {"<s>", ")", "("}:
            return k_attn_0_7_output == "("

    attn_2_5_pattern = select_closest(attn_0_7_outputs, attn_0_7_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_0_1_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_0_3_output, attn_0_7_output):
        if attn_0_3_output in {"<s>", "("}:
            return attn_0_7_output == ""
        elif attn_0_3_output in {")"}:
            return attn_0_7_output == ")"

    attn_2_6_pattern = select_closest(attn_0_7_outputs, attn_0_3_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_0_4_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(token, attn_0_0_output):
        if token in {"("}:
            return attn_0_0_output == ""
        elif token in {")"}:
            return attn_0_0_output == ")"
        elif token in {"<s>"}:
            return attn_0_0_output == "("

    attn_2_7_pattern = select_closest(attn_0_0_outputs, tokens, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_0_5_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_6_output, attn_0_1_output):
        if attn_1_6_output in {"<s>", ")", "("}:
            return attn_0_1_output == ")"

    num_attn_2_0_pattern = select(attn_0_1_outputs, attn_1_6_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_3_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_6_output, attn_0_0_output):
        if attn_1_6_output in {"<s>", "("}:
            return attn_0_0_output == ""
        elif attn_1_6_output in {")"}:
            return attn_0_0_output == "("

    num_attn_2_1_pattern = select(attn_0_0_outputs, attn_1_6_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_3_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(position, attn_1_5_output):
        if position in {
            0,
            1,
            2,
            3,
            4,
            6,
            7,
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
            return attn_1_5_output == ""
        elif position in {5}:
            return attn_1_5_output == "<s>"

    num_attn_2_2_pattern = select(attn_1_5_outputs, positions, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_2_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_6_output, num_mlp_0_1_output):
        if attn_1_6_output in {"("}:
            return num_mlp_0_1_output == 6
        elif attn_1_6_output in {")"}:
            return num_mlp_0_1_output == 16
        elif attn_1_6_output in {"<s>"}:
            return num_mlp_0_1_output == 4

    num_attn_2_3_pattern = select(
        num_mlp_0_1_outputs, attn_1_6_outputs, num_predicate_2_3
    )
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_0_0_output, attn_1_0_output):
        if attn_0_0_output in {"<s>", ")", "("}:
            return attn_1_0_output == ""

    num_attn_2_4_pattern = select(attn_1_0_outputs, attn_0_0_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_1_6_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_0_7_output, attn_1_4_output):
        if attn_0_7_output in {"<s>", ")", "("}:
            return attn_1_4_output == ""

    num_attn_2_5_pattern = select(attn_1_4_outputs, attn_0_7_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_1_3_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(num_mlp_0_0_output, attn_0_1_output):
        if num_mlp_0_0_output in {
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            15,
            16,
            17,
        }:
            return attn_0_1_output == ")"
        elif num_mlp_0_0_output in {18, 19, 14}:
            return attn_0_1_output == ""

    num_attn_2_6_pattern = select(
        attn_0_1_outputs, num_mlp_0_0_outputs, num_predicate_2_6
    )
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_0_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_0_2_output, position):
        if attn_0_2_output in {"("}:
            return position == 0
        elif attn_0_2_output in {")"}:
            return position == 14
        elif attn_0_2_output in {"<s>"}:
            return position == 1

    num_attn_2_7_pattern = select(positions, attn_0_2_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_1_0_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_1_7_output, attn_1_1_output):
        key = (attn_1_7_output, attn_1_1_output)
        return 13

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_1_7_outputs, attn_1_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_1_0_output, attn_1_2_output):
        key = (attn_1_0_output, attn_1_2_output)
        if key in {(")", ")")}:
            return 7
        elif key in {("<s>", ")")}:
            return 1
        return 6

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_1_0_outputs, attn_1_2_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_7_output, num_attn_2_1_output):
        key = (num_attn_1_7_output, num_attn_2_1_output)
        if key in {
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
            (0, 16),
            (0, 17),
            (0, 18),
            (0, 19),
            (0, 20),
            (0, 21),
            (0, 22),
            (0, 23),
            (0, 24),
            (0, 25),
            (0, 26),
            (0, 27),
            (0, 28),
            (0, 29),
            (0, 30),
            (0, 31),
            (0, 32),
            (0, 33),
            (0, 34),
            (0, 35),
            (0, 36),
            (0, 37),
            (0, 38),
            (0, 39),
            (0, 40),
            (0, 41),
            (0, 42),
            (0, 43),
            (0, 44),
            (0, 45),
            (0, 46),
            (0, 47),
            (0, 48),
            (0, 49),
            (0, 50),
            (0, 51),
            (0, 52),
            (0, 53),
            (0, 54),
            (0, 55),
            (0, 56),
            (0, 57),
            (0, 58),
            (0, 59),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (1, 13),
            (1, 14),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 20),
            (1, 21),
            (1, 22),
            (1, 23),
            (1, 24),
            (1, 25),
            (1, 26),
            (1, 27),
            (1, 28),
            (1, 29),
            (1, 30),
            (1, 31),
            (1, 32),
            (1, 33),
            (1, 34),
            (1, 35),
            (1, 36),
            (1, 37),
            (1, 38),
            (1, 39),
            (1, 40),
            (1, 41),
            (1, 42),
            (1, 43),
            (1, 44),
            (1, 45),
            (1, 46),
            (1, 47),
            (1, 48),
            (1, 49),
            (1, 50),
            (1, 51),
            (1, 52),
            (1, 53),
            (1, 54),
            (1, 55),
            (1, 56),
            (1, 57),
            (1, 58),
            (1, 59),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 10),
            (2, 11),
            (2, 12),
            (2, 13),
            (2, 14),
            (2, 15),
            (2, 16),
            (2, 17),
            (2, 18),
            (2, 19),
            (2, 20),
            (2, 21),
            (2, 22),
            (2, 23),
            (2, 24),
            (2, 25),
            (2, 26),
            (2, 27),
            (2, 28),
            (2, 29),
            (2, 30),
            (2, 31),
            (2, 32),
            (2, 33),
            (2, 34),
            (2, 35),
            (2, 36),
            (2, 37),
            (2, 38),
            (2, 39),
            (2, 40),
            (2, 41),
            (2, 42),
            (2, 43),
            (2, 44),
            (2, 45),
            (2, 46),
            (2, 47),
            (2, 48),
            (2, 49),
            (2, 50),
            (2, 51),
            (2, 52),
            (2, 53),
            (2, 54),
            (2, 55),
            (2, 56),
            (2, 57),
            (2, 58),
            (2, 59),
            (3, 7),
            (3, 8),
            (3, 9),
            (3, 10),
            (3, 11),
            (3, 12),
            (3, 13),
            (3, 14),
            (3, 15),
            (3, 16),
            (3, 17),
            (3, 18),
            (3, 19),
            (3, 20),
            (3, 21),
            (3, 22),
            (3, 23),
            (3, 24),
            (3, 25),
            (3, 26),
            (3, 27),
            (3, 28),
            (3, 29),
            (3, 30),
            (3, 31),
            (3, 32),
            (3, 33),
            (3, 34),
            (3, 35),
            (3, 36),
            (3, 37),
            (3, 38),
            (3, 39),
            (3, 40),
            (3, 41),
            (3, 42),
            (3, 43),
            (3, 44),
            (3, 45),
            (3, 46),
            (3, 47),
            (3, 48),
            (3, 49),
            (3, 50),
            (3, 51),
            (3, 52),
            (3, 53),
            (3, 54),
            (3, 55),
            (3, 56),
            (3, 57),
            (3, 58),
            (3, 59),
            (4, 8),
            (4, 9),
            (4, 10),
            (4, 11),
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 15),
            (4, 16),
            (4, 17),
            (4, 18),
            (4, 19),
            (4, 20),
            (4, 21),
            (4, 22),
            (4, 23),
            (4, 24),
            (4, 25),
            (4, 26),
            (4, 27),
            (4, 28),
            (4, 29),
            (4, 30),
            (4, 31),
            (4, 32),
            (4, 33),
            (4, 34),
            (4, 35),
            (4, 36),
            (4, 37),
            (4, 38),
            (4, 39),
            (4, 40),
            (4, 41),
            (4, 42),
            (4, 43),
            (4, 44),
            (4, 45),
            (4, 46),
            (4, 47),
            (4, 48),
            (4, 49),
            (4, 50),
            (4, 51),
            (4, 52),
            (4, 53),
            (4, 54),
            (4, 55),
            (4, 56),
            (4, 57),
            (4, 58),
            (4, 59),
            (5, 10),
            (5, 11),
            (5, 12),
            (5, 13),
            (5, 14),
            (5, 15),
            (5, 16),
            (5, 17),
            (5, 18),
            (5, 19),
            (5, 20),
            (5, 21),
            (5, 22),
            (5, 23),
            (5, 24),
            (5, 25),
            (5, 26),
            (5, 27),
            (5, 28),
            (5, 29),
            (5, 30),
            (5, 31),
            (5, 32),
            (5, 33),
            (5, 34),
            (5, 35),
            (5, 36),
            (5, 37),
            (5, 38),
            (5, 39),
            (5, 40),
            (5, 41),
            (5, 42),
            (5, 43),
            (5, 44),
            (5, 45),
            (5, 46),
            (5, 47),
            (5, 48),
            (5, 49),
            (5, 50),
            (5, 51),
            (5, 52),
            (5, 53),
            (5, 54),
            (5, 55),
            (5, 56),
            (5, 57),
            (5, 58),
            (5, 59),
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
            (6, 21),
            (6, 22),
            (6, 23),
            (6, 24),
            (6, 25),
            (6, 26),
            (6, 27),
            (6, 28),
            (6, 29),
            (6, 30),
            (6, 31),
            (6, 32),
            (6, 33),
            (6, 34),
            (6, 35),
            (6, 36),
            (6, 37),
            (6, 38),
            (6, 39),
            (6, 40),
            (6, 41),
            (6, 42),
            (6, 43),
            (6, 44),
            (6, 45),
            (6, 46),
            (6, 47),
            (6, 48),
            (6, 49),
            (6, 50),
            (6, 51),
            (6, 52),
            (6, 53),
            (6, 54),
            (6, 55),
            (6, 56),
            (6, 57),
            (6, 58),
            (6, 59),
            (7, 13),
            (7, 14),
            (7, 15),
            (7, 16),
            (7, 17),
            (7, 18),
            (7, 19),
            (7, 20),
            (7, 21),
            (7, 22),
            (7, 23),
            (7, 24),
            (7, 25),
            (7, 26),
            (7, 27),
            (7, 28),
            (7, 29),
            (7, 30),
            (7, 31),
            (7, 32),
            (7, 33),
            (7, 34),
            (7, 35),
            (7, 36),
            (7, 37),
            (7, 38),
            (7, 39),
            (7, 40),
            (7, 41),
            (7, 42),
            (7, 43),
            (7, 44),
            (7, 45),
            (7, 46),
            (7, 47),
            (7, 48),
            (7, 49),
            (7, 50),
            (7, 51),
            (7, 52),
            (7, 53),
            (7, 54),
            (7, 55),
            (7, 56),
            (7, 57),
            (7, 58),
            (7, 59),
            (8, 14),
            (8, 15),
            (8, 16),
            (8, 17),
            (8, 18),
            (8, 19),
            (8, 20),
            (8, 21),
            (8, 22),
            (8, 23),
            (8, 24),
            (8, 25),
            (8, 26),
            (8, 27),
            (8, 28),
            (8, 29),
            (8, 30),
            (8, 31),
            (8, 32),
            (8, 33),
            (8, 34),
            (8, 35),
            (8, 36),
            (8, 37),
            (8, 38),
            (8, 39),
            (8, 40),
            (8, 41),
            (8, 42),
            (8, 43),
            (8, 44),
            (8, 45),
            (8, 46),
            (8, 47),
            (8, 48),
            (8, 49),
            (8, 50),
            (8, 51),
            (8, 52),
            (8, 53),
            (8, 54),
            (8, 55),
            (8, 56),
            (8, 57),
            (8, 58),
            (8, 59),
            (9, 15),
            (9, 16),
            (9, 17),
            (9, 18),
            (9, 19),
            (9, 20),
            (9, 21),
            (9, 22),
            (9, 23),
            (9, 24),
            (9, 25),
            (9, 26),
            (9, 27),
            (9, 28),
            (9, 29),
            (9, 30),
            (9, 31),
            (9, 32),
            (9, 33),
            (9, 34),
            (9, 35),
            (9, 36),
            (9, 37),
            (9, 38),
            (9, 39),
            (9, 40),
            (9, 41),
            (9, 42),
            (9, 43),
            (9, 44),
            (9, 45),
            (9, 46),
            (9, 47),
            (9, 48),
            (9, 49),
            (9, 50),
            (9, 51),
            (9, 52),
            (9, 53),
            (9, 54),
            (9, 55),
            (9, 56),
            (9, 57),
            (9, 58),
            (9, 59),
            (10, 17),
            (10, 18),
            (10, 19),
            (10, 20),
            (10, 21),
            (10, 22),
            (10, 23),
            (10, 24),
            (10, 25),
            (10, 26),
            (10, 27),
            (10, 28),
            (10, 29),
            (10, 30),
            (10, 31),
            (10, 32),
            (10, 33),
            (10, 34),
            (10, 35),
            (10, 36),
            (10, 37),
            (10, 38),
            (10, 39),
            (10, 40),
            (10, 41),
            (10, 42),
            (10, 43),
            (10, 44),
            (10, 45),
            (10, 46),
            (10, 47),
            (10, 48),
            (10, 49),
            (10, 50),
            (10, 51),
            (10, 52),
            (10, 53),
            (10, 54),
            (10, 55),
            (10, 56),
            (10, 57),
            (10, 58),
            (10, 59),
            (11, 18),
            (11, 19),
            (11, 20),
            (11, 21),
            (11, 22),
            (11, 23),
            (11, 24),
            (11, 25),
            (11, 26),
            (11, 27),
            (11, 28),
            (11, 29),
            (11, 30),
            (11, 31),
            (11, 32),
            (11, 33),
            (11, 34),
            (11, 35),
            (11, 36),
            (11, 37),
            (11, 38),
            (11, 39),
            (11, 40),
            (11, 41),
            (11, 42),
            (11, 43),
            (11, 44),
            (11, 45),
            (11, 46),
            (11, 47),
            (11, 48),
            (11, 49),
            (11, 50),
            (11, 51),
            (11, 52),
            (11, 53),
            (11, 54),
            (11, 55),
            (11, 56),
            (11, 57),
            (11, 58),
            (11, 59),
            (12, 20),
            (12, 21),
            (12, 22),
            (12, 23),
            (12, 24),
            (12, 25),
            (12, 26),
            (12, 27),
            (12, 28),
            (12, 29),
            (12, 30),
            (12, 31),
            (12, 32),
            (12, 33),
            (12, 34),
            (12, 35),
            (12, 36),
            (12, 37),
            (12, 38),
            (12, 39),
            (12, 40),
            (12, 41),
            (12, 42),
            (12, 43),
            (12, 44),
            (12, 45),
            (12, 46),
            (12, 47),
            (12, 48),
            (12, 49),
            (12, 50),
            (12, 51),
            (12, 52),
            (12, 53),
            (12, 54),
            (12, 55),
            (12, 56),
            (12, 57),
            (12, 58),
            (12, 59),
            (13, 22),
            (13, 23),
            (13, 24),
            (13, 25),
            (13, 26),
            (13, 27),
            (13, 28),
            (13, 29),
            (13, 30),
            (13, 31),
            (13, 32),
            (13, 33),
            (13, 34),
            (13, 35),
            (13, 36),
            (13, 37),
            (13, 38),
            (13, 39),
            (13, 40),
            (13, 41),
            (13, 42),
            (13, 43),
            (13, 44),
            (13, 45),
            (13, 46),
            (13, 47),
            (13, 48),
            (13, 49),
            (13, 50),
            (13, 51),
            (13, 52),
            (13, 53),
            (13, 54),
            (13, 55),
            (13, 56),
            (13, 57),
            (13, 58),
            (13, 59),
            (14, 23),
            (14, 24),
            (14, 25),
            (14, 26),
            (14, 27),
            (14, 28),
            (14, 29),
            (14, 30),
            (14, 31),
            (14, 32),
            (14, 33),
            (14, 34),
            (14, 35),
            (14, 36),
            (14, 37),
            (14, 38),
            (14, 39),
            (14, 40),
            (14, 41),
            (14, 42),
            (14, 43),
            (14, 44),
            (14, 45),
            (14, 46),
            (14, 47),
            (14, 48),
            (14, 49),
            (14, 50),
            (14, 51),
            (14, 52),
            (14, 53),
            (14, 54),
            (14, 55),
            (14, 56),
            (14, 57),
            (14, 58),
            (14, 59),
            (15, 25),
            (15, 26),
            (15, 27),
            (15, 28),
            (15, 29),
            (15, 30),
            (15, 31),
            (15, 32),
            (15, 33),
            (15, 34),
            (15, 35),
            (15, 36),
            (15, 37),
            (15, 38),
            (15, 39),
            (15, 40),
            (15, 41),
            (15, 42),
            (15, 43),
            (15, 44),
            (15, 45),
            (15, 46),
            (15, 47),
            (15, 48),
            (15, 49),
            (15, 50),
            (15, 51),
            (15, 52),
            (15, 53),
            (15, 54),
            (15, 55),
            (15, 56),
            (15, 57),
            (15, 58),
            (15, 59),
            (16, 26),
            (16, 27),
            (16, 28),
            (16, 29),
            (16, 30),
            (16, 31),
            (16, 32),
            (16, 33),
            (16, 34),
            (16, 35),
            (16, 36),
            (16, 37),
            (16, 38),
            (16, 39),
            (16, 40),
            (16, 41),
            (16, 42),
            (16, 43),
            (16, 44),
            (16, 45),
            (16, 46),
            (16, 47),
            (16, 48),
            (16, 49),
            (16, 50),
            (16, 51),
            (16, 52),
            (16, 53),
            (16, 54),
            (16, 55),
            (16, 56),
            (16, 57),
            (16, 58),
            (16, 59),
            (17, 28),
            (17, 29),
            (17, 30),
            (17, 31),
            (17, 32),
            (17, 33),
            (17, 34),
            (17, 35),
            (17, 36),
            (17, 37),
            (17, 38),
            (17, 39),
            (17, 40),
            (17, 41),
            (17, 42),
            (17, 43),
            (17, 44),
            (17, 45),
            (17, 46),
            (17, 47),
            (17, 48),
            (17, 49),
            (17, 50),
            (17, 51),
            (17, 52),
            (17, 53),
            (17, 54),
            (17, 55),
            (17, 56),
            (17, 57),
            (17, 58),
            (17, 59),
            (18, 29),
            (18, 30),
            (18, 31),
            (18, 32),
            (18, 33),
            (18, 34),
            (18, 35),
            (18, 36),
            (18, 37),
            (18, 38),
            (18, 39),
            (18, 40),
            (18, 41),
            (18, 42),
            (18, 43),
            (18, 44),
            (18, 45),
            (18, 46),
            (18, 47),
            (18, 48),
            (18, 49),
            (18, 50),
            (18, 51),
            (18, 52),
            (18, 53),
            (18, 54),
            (18, 55),
            (18, 56),
            (18, 57),
            (18, 58),
            (18, 59),
            (19, 31),
            (19, 32),
            (19, 33),
            (19, 34),
            (19, 35),
            (19, 36),
            (19, 37),
            (19, 38),
            (19, 39),
            (19, 40),
            (19, 41),
            (19, 42),
            (19, 43),
            (19, 44),
            (19, 45),
            (19, 46),
            (19, 47),
            (19, 48),
            (19, 49),
            (19, 50),
            (19, 51),
            (19, 52),
            (19, 53),
            (19, 54),
            (19, 55),
            (19, 56),
            (19, 57),
            (19, 58),
            (19, 59),
            (20, 32),
            (20, 33),
            (20, 34),
            (20, 35),
            (20, 36),
            (20, 37),
            (20, 38),
            (20, 39),
            (20, 40),
            (20, 41),
            (20, 42),
            (20, 43),
            (20, 44),
            (20, 45),
            (20, 46),
            (20, 47),
            (20, 48),
            (20, 49),
            (20, 50),
            (20, 51),
            (20, 52),
            (20, 53),
            (20, 54),
            (20, 55),
            (20, 56),
            (20, 57),
            (20, 58),
            (20, 59),
            (21, 34),
            (21, 35),
            (21, 36),
            (21, 37),
            (21, 38),
            (21, 39),
            (21, 40),
            (21, 41),
            (21, 42),
            (21, 43),
            (21, 44),
            (21, 45),
            (21, 46),
            (21, 47),
            (21, 48),
            (21, 49),
            (21, 50),
            (21, 51),
            (21, 52),
            (21, 53),
            (21, 54),
            (21, 55),
            (21, 56),
            (21, 57),
            (21, 58),
            (21, 59),
            (22, 36),
            (22, 37),
            (22, 38),
            (22, 39),
            (22, 40),
            (22, 41),
            (22, 42),
            (22, 43),
            (22, 44),
            (22, 45),
            (22, 46),
            (22, 47),
            (22, 48),
            (22, 49),
            (22, 50),
            (22, 51),
            (22, 52),
            (22, 53),
            (22, 54),
            (22, 55),
            (22, 56),
            (22, 57),
            (22, 58),
            (22, 59),
            (23, 37),
            (23, 38),
            (23, 39),
            (23, 40),
            (23, 41),
            (23, 42),
            (23, 43),
            (23, 44),
            (23, 45),
            (23, 46),
            (23, 47),
            (23, 48),
            (23, 49),
            (23, 50),
            (23, 51),
            (23, 52),
            (23, 53),
            (23, 54),
            (23, 55),
            (23, 56),
            (23, 57),
            (23, 58),
            (23, 59),
            (24, 39),
            (24, 40),
            (24, 41),
            (24, 42),
            (24, 43),
            (24, 44),
            (24, 45),
            (24, 46),
            (24, 47),
            (24, 48),
            (24, 49),
            (24, 50),
            (24, 51),
            (24, 52),
            (24, 53),
            (24, 54),
            (24, 55),
            (24, 56),
            (24, 57),
            (24, 58),
            (24, 59),
            (25, 40),
            (25, 41),
            (25, 42),
            (25, 43),
            (25, 44),
            (25, 45),
            (25, 46),
            (25, 47),
            (25, 48),
            (25, 49),
            (25, 50),
            (25, 51),
            (25, 52),
            (25, 53),
            (25, 54),
            (25, 55),
            (25, 56),
            (25, 57),
            (25, 58),
            (25, 59),
            (26, 42),
            (26, 43),
            (26, 44),
            (26, 45),
            (26, 46),
            (26, 47),
            (26, 48),
            (26, 49),
            (26, 50),
            (26, 51),
            (26, 52),
            (26, 53),
            (26, 54),
            (26, 55),
            (26, 56),
            (26, 57),
            (26, 58),
            (26, 59),
            (27, 43),
            (27, 44),
            (27, 45),
            (27, 46),
            (27, 47),
            (27, 48),
            (27, 49),
            (27, 50),
            (27, 51),
            (27, 52),
            (27, 53),
            (27, 54),
            (27, 55),
            (27, 56),
            (27, 57),
            (27, 58),
            (27, 59),
            (28, 45),
            (28, 46),
            (28, 47),
            (28, 48),
            (28, 49),
            (28, 50),
            (28, 51),
            (28, 52),
            (28, 53),
            (28, 54),
            (28, 55),
            (28, 56),
            (28, 57),
            (28, 58),
            (28, 59),
            (29, 46),
            (29, 47),
            (29, 48),
            (29, 49),
            (29, 50),
            (29, 51),
            (29, 52),
            (29, 53),
            (29, 54),
            (29, 55),
            (29, 56),
            (29, 57),
            (29, 58),
            (29, 59),
            (30, 48),
            (30, 49),
            (30, 50),
            (30, 51),
            (30, 52),
            (30, 53),
            (30, 54),
            (30, 55),
            (30, 56),
            (30, 57),
            (30, 58),
            (30, 59),
            (31, 50),
            (31, 51),
            (31, 52),
            (31, 53),
            (31, 54),
            (31, 55),
            (31, 56),
            (31, 57),
            (31, 58),
            (31, 59),
            (32, 51),
            (32, 52),
            (32, 53),
            (32, 54),
            (32, 55),
            (32, 56),
            (32, 57),
            (32, 58),
            (32, 59),
            (33, 53),
            (33, 54),
            (33, 55),
            (33, 56),
            (33, 57),
            (33, 58),
            (33, 59),
            (34, 54),
            (34, 55),
            (34, 56),
            (34, 57),
            (34, 58),
            (34, 59),
            (35, 56),
            (35, 57),
            (35, 58),
            (35, 59),
            (36, 57),
            (36, 58),
            (36, 59),
            (37, 59),
        }:
            return 17
        elif key in {
            (10, 10),
            (11, 11),
            (12, 11),
            (12, 12),
            (13, 12),
            (13, 13),
            (14, 12),
            (14, 13),
            (14, 14),
            (15, 13),
            (15, 14),
            (15, 15),
            (15, 16),
            (16, 14),
            (16, 15),
            (16, 16),
            (16, 17),
            (17, 14),
            (17, 15),
            (17, 16),
            (17, 17),
            (17, 18),
            (18, 15),
            (18, 16),
            (18, 17),
            (18, 18),
            (18, 19),
            (19, 15),
            (19, 16),
            (19, 17),
            (19, 18),
            (19, 19),
            (19, 20),
            (20, 16),
            (20, 17),
            (20, 18),
            (20, 19),
            (20, 20),
            (20, 21),
            (21, 16),
            (21, 17),
            (21, 18),
            (21, 19),
            (21, 20),
            (21, 21),
            (21, 22),
            (22, 17),
            (22, 18),
            (22, 19),
            (22, 20),
            (22, 21),
            (22, 22),
            (22, 23),
            (22, 24),
            (23, 17),
            (23, 18),
            (23, 19),
            (23, 20),
            (23, 21),
            (23, 22),
            (23, 23),
            (23, 24),
            (23, 25),
            (24, 18),
            (24, 19),
            (24, 20),
            (24, 21),
            (24, 22),
            (24, 23),
            (24, 24),
            (24, 25),
            (24, 26),
            (25, 18),
            (25, 19),
            (25, 20),
            (25, 21),
            (25, 22),
            (25, 23),
            (25, 24),
            (25, 25),
            (25, 26),
            (25, 27),
            (26, 19),
            (26, 20),
            (26, 21),
            (26, 22),
            (26, 23),
            (26, 24),
            (26, 25),
            (26, 26),
            (26, 27),
            (26, 28),
            (27, 19),
            (27, 20),
            (27, 21),
            (27, 22),
            (27, 23),
            (27, 24),
            (27, 25),
            (27, 26),
            (27, 27),
            (27, 28),
            (27, 29),
            (28, 20),
            (28, 21),
            (28, 22),
            (28, 23),
            (28, 24),
            (28, 25),
            (28, 26),
            (28, 27),
            (28, 28),
            (28, 29),
            (28, 30),
            (29, 21),
            (29, 22),
            (29, 23),
            (29, 24),
            (29, 25),
            (29, 26),
            (29, 27),
            (29, 28),
            (29, 29),
            (29, 30),
            (29, 31),
            (29, 32),
            (30, 21),
            (30, 22),
            (30, 23),
            (30, 24),
            (30, 25),
            (30, 26),
            (30, 27),
            (30, 28),
            (30, 29),
            (30, 30),
            (30, 31),
            (30, 32),
            (30, 33),
            (31, 22),
            (31, 23),
            (31, 24),
            (31, 25),
            (31, 26),
            (31, 27),
            (31, 28),
            (31, 29),
            (31, 30),
            (31, 31),
            (31, 32),
            (31, 33),
            (31, 34),
            (32, 22),
            (32, 23),
            (32, 24),
            (32, 25),
            (32, 26),
            (32, 27),
            (32, 28),
            (32, 29),
            (32, 30),
            (32, 31),
            (32, 32),
            (32, 33),
            (32, 34),
            (32, 35),
            (33, 23),
            (33, 24),
            (33, 25),
            (33, 26),
            (33, 27),
            (33, 28),
            (33, 29),
            (33, 30),
            (33, 31),
            (33, 32),
            (33, 33),
            (33, 34),
            (33, 35),
            (33, 36),
            (34, 23),
            (34, 24),
            (34, 25),
            (34, 26),
            (34, 27),
            (34, 28),
            (34, 29),
            (34, 30),
            (34, 31),
            (34, 32),
            (34, 33),
            (34, 34),
            (34, 35),
            (34, 36),
            (34, 37),
            (35, 24),
            (35, 25),
            (35, 26),
            (35, 27),
            (35, 28),
            (35, 29),
            (35, 30),
            (35, 31),
            (35, 32),
            (35, 33),
            (35, 34),
            (35, 35),
            (35, 36),
            (35, 37),
            (35, 38),
            (36, 24),
            (36, 25),
            (36, 26),
            (36, 27),
            (36, 28),
            (36, 29),
            (36, 30),
            (36, 31),
            (36, 32),
            (36, 33),
            (36, 34),
            (36, 35),
            (36, 36),
            (36, 37),
            (36, 38),
            (36, 39),
            (37, 25),
            (37, 26),
            (37, 27),
            (37, 28),
            (37, 29),
            (37, 30),
            (37, 31),
            (37, 32),
            (37, 33),
            (37, 34),
            (37, 35),
            (37, 36),
            (37, 37),
            (37, 38),
            (37, 39),
            (37, 40),
            (37, 41),
            (38, 25),
            (38, 26),
            (38, 27),
            (38, 28),
            (38, 29),
            (38, 30),
            (38, 31),
            (38, 32),
            (38, 33),
            (38, 34),
            (38, 35),
            (38, 36),
            (38, 37),
            (38, 38),
            (38, 39),
            (38, 40),
            (38, 41),
            (38, 42),
            (39, 26),
            (39, 27),
            (39, 28),
            (39, 29),
            (39, 30),
            (39, 31),
            (39, 32),
            (39, 33),
            (39, 34),
            (39, 35),
            (39, 36),
            (39, 37),
            (39, 38),
            (39, 39),
            (39, 40),
            (39, 41),
            (39, 42),
            (39, 43),
            (40, 27),
            (40, 28),
            (40, 29),
            (40, 30),
            (40, 31),
            (40, 32),
            (40, 33),
            (40, 34),
            (40, 35),
            (40, 36),
            (40, 37),
            (40, 38),
            (40, 39),
            (40, 40),
            (40, 41),
            (40, 42),
            (40, 43),
            (40, 44),
            (41, 27),
            (41, 28),
            (41, 29),
            (41, 30),
            (41, 31),
            (41, 32),
            (41, 33),
            (41, 34),
            (41, 35),
            (41, 36),
            (41, 37),
            (41, 38),
            (41, 39),
            (41, 40),
            (41, 41),
            (41, 42),
            (41, 43),
            (41, 44),
            (41, 45),
            (42, 28),
            (42, 29),
            (42, 30),
            (42, 31),
            (42, 32),
            (42, 33),
            (42, 34),
            (42, 35),
            (42, 36),
            (42, 37),
            (42, 38),
            (42, 39),
            (42, 40),
            (42, 41),
            (42, 42),
            (42, 43),
            (42, 44),
            (42, 45),
            (42, 46),
            (43, 28),
            (43, 29),
            (43, 30),
            (43, 31),
            (43, 32),
            (43, 33),
            (43, 34),
            (43, 35),
            (43, 36),
            (43, 37),
            (43, 38),
            (43, 39),
            (43, 40),
            (43, 41),
            (43, 42),
            (43, 43),
            (43, 44),
            (43, 45),
            (43, 46),
            (43, 47),
            (44, 29),
            (44, 30),
            (44, 31),
            (44, 32),
            (44, 33),
            (44, 34),
            (44, 35),
            (44, 36),
            (44, 37),
            (44, 38),
            (44, 39),
            (44, 40),
            (44, 41),
            (44, 42),
            (44, 43),
            (44, 44),
            (44, 45),
            (44, 46),
            (44, 47),
            (44, 48),
            (45, 29),
            (45, 30),
            (45, 31),
            (45, 32),
            (45, 33),
            (45, 34),
            (45, 35),
            (45, 36),
            (45, 37),
            (45, 38),
            (45, 39),
            (45, 40),
            (45, 41),
            (45, 42),
            (45, 43),
            (45, 44),
            (45, 45),
            (45, 46),
            (45, 47),
            (45, 48),
            (45, 49),
            (45, 50),
            (46, 30),
            (46, 31),
            (46, 32),
            (46, 33),
            (46, 34),
            (46, 35),
            (46, 36),
            (46, 37),
            (46, 38),
            (46, 39),
            (46, 40),
            (46, 41),
            (46, 42),
            (46, 43),
            (46, 44),
            (46, 45),
            (46, 46),
            (46, 47),
            (46, 48),
            (46, 49),
            (46, 50),
            (46, 51),
            (47, 30),
            (47, 31),
            (47, 32),
            (47, 33),
            (47, 34),
            (47, 35),
            (47, 36),
            (47, 37),
            (47, 38),
            (47, 39),
            (47, 40),
            (47, 41),
            (47, 42),
            (47, 43),
            (47, 44),
            (47, 45),
            (47, 46),
            (47, 47),
            (47, 48),
            (47, 49),
            (47, 50),
            (47, 51),
            (47, 52),
            (48, 31),
            (48, 32),
            (48, 33),
            (48, 34),
            (48, 35),
            (48, 36),
            (48, 37),
            (48, 38),
            (48, 39),
            (48, 40),
            (48, 41),
            (48, 42),
            (48, 43),
            (48, 44),
            (48, 45),
            (48, 46),
            (48, 47),
            (48, 48),
            (48, 49),
            (48, 50),
            (48, 51),
            (48, 52),
            (48, 53),
            (49, 31),
            (49, 32),
            (49, 33),
            (49, 34),
            (49, 35),
            (49, 36),
            (49, 37),
            (49, 38),
            (49, 39),
            (49, 40),
            (49, 41),
            (49, 42),
            (49, 43),
            (49, 44),
            (49, 45),
            (49, 46),
            (49, 47),
            (49, 48),
            (49, 49),
            (49, 50),
            (49, 51),
            (49, 52),
            (49, 53),
            (49, 54),
            (50, 32),
            (50, 33),
            (50, 34),
            (50, 35),
            (50, 36),
            (50, 37),
            (50, 38),
            (50, 39),
            (50, 40),
            (50, 41),
            (50, 42),
            (50, 43),
            (50, 44),
            (50, 45),
            (50, 46),
            (50, 47),
            (50, 48),
            (50, 49),
            (50, 50),
            (50, 51),
            (50, 52),
            (50, 53),
            (50, 54),
            (50, 55),
            (51, 33),
            (51, 34),
            (51, 35),
            (51, 36),
            (51, 37),
            (51, 38),
            (51, 39),
            (51, 40),
            (51, 41),
            (51, 42),
            (51, 43),
            (51, 44),
            (51, 45),
            (51, 46),
            (51, 47),
            (51, 48),
            (51, 49),
            (51, 50),
            (51, 51),
            (51, 52),
            (51, 53),
            (51, 54),
            (51, 55),
            (51, 56),
            (52, 33),
            (52, 34),
            (52, 35),
            (52, 36),
            (52, 37),
            (52, 38),
            (52, 39),
            (52, 40),
            (52, 41),
            (52, 42),
            (52, 43),
            (52, 44),
            (52, 45),
            (52, 46),
            (52, 47),
            (52, 48),
            (52, 49),
            (52, 50),
            (52, 51),
            (52, 52),
            (52, 53),
            (52, 54),
            (52, 55),
            (52, 56),
            (52, 57),
            (52, 58),
            (53, 34),
            (53, 35),
            (53, 36),
            (53, 37),
            (53, 38),
            (53, 39),
            (53, 40),
            (53, 41),
            (53, 42),
            (53, 43),
            (53, 44),
            (53, 45),
            (53, 46),
            (53, 47),
            (53, 48),
            (53, 49),
            (53, 50),
            (53, 51),
            (53, 52),
            (53, 53),
            (53, 54),
            (53, 55),
            (53, 56),
            (53, 57),
            (53, 58),
            (53, 59),
            (54, 34),
            (54, 35),
            (54, 36),
            (54, 37),
            (54, 38),
            (54, 39),
            (54, 40),
            (54, 41),
            (54, 42),
            (54, 43),
            (54, 44),
            (54, 45),
            (54, 46),
            (54, 47),
            (54, 48),
            (54, 49),
            (54, 50),
            (54, 51),
            (54, 52),
            (54, 53),
            (54, 54),
            (54, 55),
            (54, 56),
            (54, 57),
            (54, 58),
            (54, 59),
            (55, 35),
            (55, 36),
            (55, 37),
            (55, 38),
            (55, 39),
            (55, 40),
            (55, 41),
            (55, 42),
            (55, 43),
            (55, 44),
            (55, 45),
            (55, 46),
            (55, 47),
            (55, 48),
            (55, 49),
            (55, 50),
            (55, 51),
            (55, 52),
            (55, 53),
            (55, 54),
            (55, 55),
            (55, 56),
            (55, 57),
            (55, 58),
            (55, 59),
            (56, 35),
            (56, 36),
            (56, 37),
            (56, 38),
            (56, 39),
            (56, 40),
            (56, 41),
            (56, 42),
            (56, 43),
            (56, 44),
            (56, 45),
            (56, 46),
            (56, 47),
            (56, 48),
            (56, 49),
            (56, 50),
            (56, 51),
            (56, 52),
            (56, 53),
            (56, 54),
            (56, 55),
            (56, 56),
            (56, 57),
            (56, 58),
            (56, 59),
            (57, 36),
            (57, 37),
            (57, 38),
            (57, 39),
            (57, 40),
            (57, 41),
            (57, 42),
            (57, 43),
            (57, 44),
            (57, 45),
            (57, 46),
            (57, 47),
            (57, 48),
            (57, 49),
            (57, 50),
            (57, 51),
            (57, 52),
            (57, 53),
            (57, 54),
            (57, 55),
            (57, 56),
            (57, 57),
            (57, 58),
            (57, 59),
            (58, 36),
            (58, 37),
            (58, 38),
            (58, 39),
            (58, 40),
            (58, 41),
            (58, 42),
            (58, 43),
            (58, 44),
            (58, 45),
            (58, 46),
            (58, 47),
            (58, 48),
            (58, 49),
            (58, 50),
            (58, 51),
            (58, 52),
            (58, 53),
            (58, 54),
            (58, 55),
            (58, 56),
            (58, 57),
            (58, 58),
            (58, 59),
            (59, 37),
            (59, 38),
            (59, 39),
            (59, 40),
            (59, 41),
            (59, 42),
            (59, 43),
            (59, 44),
            (59, 45),
            (59, 46),
            (59, 47),
            (59, 48),
            (59, 49),
            (59, 50),
            (59, 51),
            (59, 52),
            (59, 53),
            (59, 54),
            (59, 55),
            (59, 56),
            (59, 57),
            (59, 58),
            (59, 59),
        }:
            return 3
        elif key in {
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 2),
            (2, 3),
            (2, 4),
            (3, 4),
            (3, 5),
            (4, 5),
            (4, 6),
            (5, 6),
            (5, 7),
            (5, 8),
            (6, 7),
            (6, 8),
            (6, 9),
            (7, 8),
            (7, 9),
            (7, 10),
            (7, 11),
            (8, 9),
            (8, 10),
            (8, 11),
            (8, 12),
            (9, 11),
            (9, 12),
            (9, 13),
            (9, 14),
            (10, 12),
            (10, 13),
            (10, 14),
            (10, 15),
            (11, 13),
            (11, 14),
            (11, 15),
            (11, 16),
            (11, 17),
            (12, 14),
            (12, 15),
            (12, 16),
            (12, 17),
            (12, 18),
            (13, 15),
            (13, 16),
            (13, 17),
            (13, 18),
            (13, 19),
            (13, 20),
            (14, 16),
            (14, 17),
            (14, 18),
            (14, 19),
            (14, 20),
            (14, 21),
            (15, 17),
            (15, 18),
            (15, 19),
            (15, 20),
            (15, 21),
            (15, 22),
            (16, 19),
            (16, 20),
            (16, 21),
            (16, 22),
            (16, 23),
            (16, 24),
            (17, 20),
            (17, 21),
            (17, 22),
            (17, 23),
            (17, 24),
            (17, 25),
            (18, 21),
            (18, 22),
            (18, 23),
            (18, 24),
            (18, 25),
            (18, 26),
            (18, 27),
            (19, 22),
            (19, 23),
            (19, 24),
            (19, 25),
            (19, 26),
            (19, 27),
            (19, 28),
            (20, 23),
            (20, 24),
            (20, 25),
            (20, 26),
            (20, 27),
            (20, 28),
            (20, 29),
            (21, 24),
            (21, 25),
            (21, 26),
            (21, 27),
            (21, 28),
            (21, 29),
            (21, 30),
            (21, 31),
            (22, 25),
            (22, 26),
            (22, 27),
            (22, 28),
            (22, 29),
            (22, 30),
            (22, 31),
            (22, 32),
            (23, 27),
            (23, 28),
            (23, 29),
            (23, 30),
            (23, 31),
            (23, 32),
            (23, 33),
            (23, 34),
            (24, 28),
            (24, 29),
            (24, 30),
            (24, 31),
            (24, 32),
            (24, 33),
            (24, 34),
            (24, 35),
            (25, 29),
            (25, 30),
            (25, 31),
            (25, 32),
            (25, 33),
            (25, 34),
            (25, 35),
            (25, 36),
            (25, 37),
            (26, 30),
            (26, 31),
            (26, 32),
            (26, 33),
            (26, 34),
            (26, 35),
            (26, 36),
            (26, 37),
            (26, 38),
            (27, 31),
            (27, 32),
            (27, 33),
            (27, 34),
            (27, 35),
            (27, 36),
            (27, 37),
            (27, 38),
            (27, 39),
            (28, 32),
            (28, 33),
            (28, 34),
            (28, 35),
            (28, 36),
            (28, 37),
            (28, 38),
            (28, 39),
            (28, 40),
            (28, 41),
            (29, 34),
            (29, 35),
            (29, 36),
            (29, 37),
            (29, 38),
            (29, 39),
            (29, 40),
            (29, 41),
            (29, 42),
            (30, 35),
            (30, 36),
            (30, 37),
            (30, 38),
            (30, 39),
            (30, 40),
            (30, 41),
            (30, 42),
            (30, 43),
            (30, 44),
            (31, 36),
            (31, 37),
            (31, 38),
            (31, 39),
            (31, 40),
            (31, 41),
            (31, 42),
            (31, 43),
            (31, 44),
            (31, 45),
            (32, 37),
            (32, 38),
            (32, 39),
            (32, 40),
            (32, 41),
            (32, 42),
            (32, 43),
            (32, 44),
            (32, 45),
            (32, 46),
            (33, 38),
            (33, 39),
            (33, 40),
            (33, 41),
            (33, 42),
            (33, 43),
            (33, 44),
            (33, 45),
            (33, 46),
            (33, 47),
            (33, 48),
            (34, 39),
            (34, 40),
            (34, 41),
            (34, 42),
            (34, 43),
            (34, 44),
            (34, 45),
            (34, 46),
            (34, 47),
            (34, 48),
            (34, 49),
            (35, 40),
            (35, 41),
            (35, 42),
            (35, 43),
            (35, 44),
            (35, 45),
            (35, 46),
            (35, 47),
            (35, 48),
            (35, 49),
            (35, 50),
            (35, 51),
            (36, 42),
            (36, 43),
            (36, 44),
            (36, 45),
            (36, 46),
            (36, 47),
            (36, 48),
            (36, 49),
            (36, 50),
            (36, 51),
            (36, 52),
            (37, 43),
            (37, 44),
            (37, 45),
            (37, 46),
            (37, 47),
            (37, 48),
            (37, 49),
            (37, 50),
            (37, 51),
            (37, 52),
            (37, 53),
            (37, 54),
            (38, 44),
            (38, 45),
            (38, 46),
            (38, 47),
            (38, 48),
            (38, 49),
            (38, 50),
            (38, 51),
            (38, 52),
            (38, 53),
            (38, 54),
            (38, 55),
            (39, 45),
            (39, 46),
            (39, 47),
            (39, 48),
            (39, 49),
            (39, 50),
            (39, 51),
            (39, 52),
            (39, 53),
            (39, 54),
            (39, 55),
            (39, 56),
            (40, 46),
            (40, 47),
            (40, 48),
            (40, 49),
            (40, 50),
            (40, 51),
            (40, 52),
            (40, 53),
            (40, 54),
            (40, 55),
            (40, 56),
            (40, 57),
            (40, 58),
            (41, 47),
            (41, 48),
            (41, 49),
            (41, 50),
            (41, 51),
            (41, 52),
            (41, 53),
            (41, 54),
            (41, 55),
            (41, 56),
            (41, 57),
            (41, 58),
            (41, 59),
            (42, 48),
            (42, 49),
            (42, 50),
            (42, 51),
            (42, 52),
            (42, 53),
            (42, 54),
            (42, 55),
            (42, 56),
            (42, 57),
            (42, 58),
            (42, 59),
            (43, 50),
            (43, 51),
            (43, 52),
            (43, 53),
            (43, 54),
            (43, 55),
            (43, 56),
            (43, 57),
            (43, 58),
            (43, 59),
            (44, 51),
            (44, 52),
            (44, 53),
            (44, 54),
            (44, 55),
            (44, 56),
            (44, 57),
            (44, 58),
            (44, 59),
            (45, 52),
            (45, 53),
            (45, 54),
            (45, 55),
            (45, 56),
            (45, 57),
            (45, 58),
            (45, 59),
            (46, 53),
            (46, 54),
            (46, 55),
            (46, 56),
            (46, 57),
            (46, 58),
            (46, 59),
            (47, 54),
            (47, 55),
            (47, 56),
            (47, 57),
            (47, 58),
            (47, 59),
            (48, 55),
            (48, 56),
            (48, 57),
            (48, 58),
            (48, 59),
            (49, 57),
            (49, 58),
            (49, 59),
            (50, 58),
            (50, 59),
            (51, 59),
        }:
            return 10
        elif key in {
            (12, 19),
            (13, 21),
            (14, 22),
            (15, 23),
            (15, 24),
            (16, 25),
            (17, 26),
            (17, 27),
            (18, 28),
            (19, 29),
            (19, 30),
            (20, 30),
            (20, 31),
            (21, 32),
            (21, 33),
            (22, 33),
            (22, 34),
            (22, 35),
            (23, 35),
            (23, 36),
            (24, 36),
            (24, 37),
            (24, 38),
            (25, 38),
            (25, 39),
            (26, 39),
            (26, 40),
            (26, 41),
            (27, 40),
            (27, 41),
            (27, 42),
            (28, 42),
            (28, 43),
            (28, 44),
            (29, 43),
            (29, 44),
            (29, 45),
            (30, 45),
            (30, 46),
            (30, 47),
            (31, 46),
            (31, 47),
            (31, 48),
            (31, 49),
            (32, 47),
            (32, 48),
            (32, 49),
            (32, 50),
            (33, 49),
            (33, 50),
            (33, 51),
            (33, 52),
            (34, 50),
            (34, 51),
            (34, 52),
            (34, 53),
            (35, 52),
            (35, 53),
            (35, 54),
            (35, 55),
            (36, 53),
            (36, 54),
            (36, 55),
            (36, 56),
            (37, 55),
            (37, 56),
            (37, 57),
            (37, 58),
            (38, 56),
            (38, 57),
            (38, 58),
            (38, 59),
            (39, 57),
            (39, 58),
            (39, 59),
            (40, 59),
        }:
            return 6
        elif key in {
            (2, 2),
            (9, 10),
            (10, 11),
            (11, 12),
            (16, 18),
            (17, 19),
            (18, 20),
            (19, 21),
            (20, 22),
            (23, 26),
            (24, 27),
            (25, 28),
            (26, 29),
            (27, 30),
            (29, 33),
            (30, 34),
            (31, 35),
            (32, 36),
            (33, 37),
            (36, 41),
            (37, 42),
            (38, 43),
            (39, 44),
            (40, 45),
            (43, 49),
            (44, 50),
            (45, 51),
            (46, 52),
            (49, 56),
            (50, 57),
            (51, 58),
        }:
            return 16
        elif key in {
            (21, 23),
            (28, 31),
            (34, 38),
            (35, 39),
            (36, 40),
            (41, 46),
            (42, 47),
            (43, 48),
            (44, 49),
            (47, 53),
            (48, 54),
            (49, 55),
            (50, 56),
            (51, 57),
        }:
            return 19
        elif key in {(52, 59)}:
            return 8
        return 13

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_5_output, num_attn_1_0_output):
        key = (num_attn_2_5_output, num_attn_1_0_output)
        return 2

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_5_outputs, num_attn_1_0_outputs)
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


print(run(["<s>", "(", ")", ")", ")", "(", ")", "(", "(", ")"]))
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
        "output/length/rasp/double_hist/trainlength10/s2/double_hist_weights.csv",
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
    def predicate_0_0(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "3"
        elif q_token in {"1"}:
            return k_token == "2"
        elif q_token in {"2"}:
            return k_token == "0"
        elif q_token in {"3", "5"}:
            return k_token == "1"
        elif q_token in {"4"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == "4"

    attn_0_0_pattern = select_closest(tokens, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"<s>", "1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == ""
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"

    attn_0_1_pattern = select_closest(tokens, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_token, k_token):
        if q_token in {"0", "5"}:
            return k_token == "2"
        elif q_token in {"4", "1"}:
            return k_token == "0"
        elif q_token in {"2"}:
            return k_token == "1"
        elif q_token in {"3"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == "<s>"

    attn_0_2_pattern = select_closest(tokens, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, positions)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_token, k_token):
        if q_token in {"0", "1"}:
            return k_token == "5"
        elif q_token in {"2"}:
            return k_token == "3"
        elif q_token in {"3", "5", "4"}:
            return k_token == "2"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_0_3_pattern = select_closest(tokens, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
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
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0, 17}:
            return k_position == 8
        elif q_position in {1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 18, 19}:
            return k_position == 7
        elif q_position in {8}:
            return k_position == 0
        elif q_position in {9}:
            return k_position == 9

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0, 10, 11, 12, 13, 15, 16, 18, 19}:
            return k_position == 8
        elif q_position in {1, 2, 3, 4, 5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 3
        elif q_position in {7}:
            return k_position == 4
        elif q_position in {8}:
            return k_position == 5
        elif q_position in {9}:
            return k_position == 0
        elif q_position in {14}:
            return k_position == 17
        elif q_position in {17}:
            return k_position == 19

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
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
    def mlp_0_0(attn_0_1_output, attn_0_0_output):
        key = (attn_0_1_output, attn_0_0_output)
        return 14

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_0_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_0_output, attn_0_1_output):
        key = (attn_0_0_output, attn_0_1_output)
        if key in {
            ("0", "0"),
            ("0", "2"),
            ("0", "3"),
            ("0", "4"),
            ("0", "5"),
            ("0", "<s>"),
            ("1", "3"),
            ("1", "5"),
            ("2", "3"),
            ("2", "5"),
            ("3", "0"),
            ("3", "3"),
            ("3", "5"),
            ("3", "<s>"),
            ("4", "0"),
            ("4", "3"),
            ("4", "5"),
            ("5", "0"),
            ("5", "3"),
            ("5", "5"),
            ("<s>", "3"),
            ("<s>", "5"),
        }:
            return 9
        return 6

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_1_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_3_output):
        key = num_attn_0_3_output
        if key in {0, 1}:
            return 10
        elif key in {2}:
            return 14
        return 13

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_3_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_2_output):
        key = num_attn_0_2_output
        return 10

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 10, 11, 12, 15, 17, 18, 19}:
            return k_position == 6
        elif q_position in {1, 2, 4, 9}:
            return k_position == 9
        elif q_position in {3, 5, 6, 7}:
            return k_position == 8
        elif q_position in {8}:
            return k_position == 3
        elif q_position in {13}:
            return k_position == 4
        elif q_position in {14}:
            return k_position == 5
        elif q_position in {16}:
            return k_position == 15

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, positions)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 17
        elif q_position in {1, 5, 9, 7}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {19, 3}:
            return k_position == 7
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 8
        elif q_position in {8, 17, 11}:
            return k_position == 6
        elif q_position in {10, 13, 15, 16, 18}:
            return k_position == 9
        elif q_position in {12, 14}:
            return k_position == 15

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_1_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(token, mlp_0_1_output):
        if token in {"0"}:
            return mlp_0_1_output == 15
        elif token in {"3", "<s>", "5", "1"}:
            return mlp_0_1_output == 10
        elif token in {"2", "4"}:
            return mlp_0_1_output == 17

    attn_1_2_pattern = select_closest(mlp_0_1_outputs, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, mlp_0_1_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_2_output, position):
        if attn_0_2_output in {0}:
            return position == 4
        elif attn_0_2_output in {1, 2, 3, 4, 15, 16, 19}:
            return position == 5
        elif attn_0_2_output in {5}:
            return position == 6
        elif attn_0_2_output in {6}:
            return position == 1
        elif attn_0_2_output in {7}:
            return position == 8
        elif attn_0_2_output in {8, 13}:
            return position == 2
        elif attn_0_2_output in {9}:
            return position == 3
        elif attn_0_2_output in {10}:
            return position == 17
        elif attn_0_2_output in {11}:
            return position == 18
        elif attn_0_2_output in {12, 14}:
            return position == 9
        elif attn_0_2_output in {17}:
            return position == 12
        elif attn_0_2_output in {18}:
            return position == 14

    attn_1_3_pattern = select_closest(positions, attn_0_2_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, positions)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(token, attn_0_3_output):
        if token in {"0"}:
            return attn_0_3_output == "0"
        elif token in {"1"}:
            return attn_0_3_output == "1"
        elif token in {"2"}:
            return attn_0_3_output == "2"
        elif token in {"3", "<s>"}:
            return attn_0_3_output == "3"
        elif token in {"4"}:
            return attn_0_3_output == "4"
        elif token in {"5"}:
            return attn_0_3_output == "5"

    num_attn_1_0_pattern = select(attn_0_3_outputs, tokens, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_3_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(q_num_mlp_0_0_output, k_num_mlp_0_0_output):
        if q_num_mlp_0_0_output in {0, 10, 17, 18, 19}:
            return k_num_mlp_0_0_output == 14
        elif q_num_mlp_0_0_output in {1, 3, 4, 5, 6, 7, 12, 15, 16}:
            return k_num_mlp_0_0_output == 13
        elif q_num_mlp_0_0_output in {2}:
            return k_num_mlp_0_0_output == 2
        elif q_num_mlp_0_0_output in {8, 11}:
            return k_num_mlp_0_0_output == 9
        elif q_num_mlp_0_0_output in {9, 13, 14}:
            return k_num_mlp_0_0_output == 10

    num_attn_1_1_pattern = select(
        num_mlp_0_0_outputs, num_mlp_0_0_outputs, num_predicate_1_1
    )
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_3_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(q_num_mlp_0_0_output, k_num_mlp_0_0_output):
        if q_num_mlp_0_0_output in {
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            9,
            10,
            11,
            12,
            13,
            16,
            17,
            18,
        }:
            return k_num_mlp_0_0_output == 13
        elif q_num_mlp_0_0_output in {8}:
            return k_num_mlp_0_0_output == 15
        elif q_num_mlp_0_0_output in {14}:
            return k_num_mlp_0_0_output == 10
        elif q_num_mlp_0_0_output in {15}:
            return k_num_mlp_0_0_output == 16
        elif q_num_mlp_0_0_output in {19}:
            return k_num_mlp_0_0_output == 2

    num_attn_1_2_pattern = select(
        num_mlp_0_0_outputs, num_mlp_0_0_outputs, num_predicate_1_2
    )
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, ones)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(q_num_mlp_0_0_output, k_num_mlp_0_0_output):
        if q_num_mlp_0_0_output in {0, 8, 10, 11, 15, 16, 17, 18, 19}:
            return k_num_mlp_0_0_output == 10
        elif q_num_mlp_0_0_output in {1, 3, 4, 5, 7, 9, 12, 14}:
            return k_num_mlp_0_0_output == 14
        elif q_num_mlp_0_0_output in {2}:
            return k_num_mlp_0_0_output == 8
        elif q_num_mlp_0_0_output in {6}:
            return k_num_mlp_0_0_output == 9
        elif q_num_mlp_0_0_output in {13}:
            return k_num_mlp_0_0_output == 13

    num_attn_1_3_pattern = select(
        num_mlp_0_0_outputs, num_mlp_0_0_outputs, num_predicate_1_3
    )
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_0_output):
        key = attn_0_0_output
        if key in {"", "1", "2", "3", "4"}:
            return 7
        return 3

    mlp_1_0_outputs = [mlp_1_0(k0) for k0 in attn_0_0_outputs]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_2_output, attn_1_2_output):
        key = (attn_0_2_output, attn_1_2_output)
        return 17

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_1_2_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_1_output, num_attn_0_2_output):
        key = (num_attn_0_1_output, num_attn_0_2_output)
        return 17

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_1_output, num_attn_0_0_output):
        key = (num_attn_0_1_output, num_attn_0_0_output)
        return 18

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_position, k_position):
        if q_position in {0, 2, 7, 10, 11, 12, 15, 16, 17, 18, 19}:
            return k_position == 6
        elif q_position in {8, 1}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 9
        elif q_position in {9, 6}:
            return k_position == 3
        elif q_position in {13, 14}:
            return k_position == 8

    attn_2_0_pattern = select_closest(positions, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, mlp_0_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_position, k_position):
        if q_position in {0, 7, 10, 11, 12, 15, 16, 17, 18, 19}:
            return k_position == 6
        elif q_position in {8, 1, 2, 4}:
            return k_position == 5
        elif q_position in {3, 13}:
            return k_position == 9
        elif q_position in {5}:
            return k_position == 2
        elif q_position in {6}:
            return k_position == 3
        elif q_position in {9}:
            return k_position == 4
        elif q_position in {14}:
            return k_position == 11

    attn_2_1_pattern = select_closest(positions, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(position, num_mlp_0_0_output):
        if position in {0, 1, 2, 4, 5, 6, 13}:
            return num_mlp_0_0_output == 13
        elif position in {3}:
            return num_mlp_0_0_output == 8
        elif position in {7}:
            return num_mlp_0_0_output == 14
        elif position in {8}:
            return num_mlp_0_0_output == 10
        elif position in {9}:
            return num_mlp_0_0_output == 1
        elif position in {18, 19, 10, 11}:
            return num_mlp_0_0_output == 7
        elif position in {17, 12, 15}:
            return num_mlp_0_0_output == 6
        elif position in {14}:
            return num_mlp_0_0_output == 16
        elif position in {16}:
            return num_mlp_0_0_output == 9

    attn_2_2_pattern = select_closest(num_mlp_0_0_outputs, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_token, k_token):
        if q_token in {"2", "0"}:
            return k_token == "5"
        elif q_token in {"3", "1"}:
            return k_token == "2"
        elif q_token in {"4"}:
            return k_token == "0"
        elif q_token in {"5"}:
            return k_token == "1"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_3_pattern = select_closest(tokens, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, num_mlp_0_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_0_3_output, mlp_0_0_output):
        if attn_0_3_output in {"0"}:
            return mlp_0_0_output == 3
        elif attn_0_3_output in {"1"}:
            return mlp_0_0_output == 1
        elif attn_0_3_output in {"2"}:
            return mlp_0_0_output == 9
        elif attn_0_3_output in {"3", "5"}:
            return mlp_0_0_output == 18
        elif attn_0_3_output in {"4"}:
            return mlp_0_0_output == 13
        elif attn_0_3_output in {"<s>"}:
            return mlp_0_0_output == 12

    num_attn_2_0_pattern = select(mlp_0_0_outputs, attn_0_3_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_1_output, position):
        if attn_1_1_output in {0, 2, 3}:
            return position == 16
        elif attn_1_1_output in {1, 5, 15}:
            return position == 12
        elif attn_1_1_output in {4}:
            return position == 13
        elif attn_1_1_output in {9, 12, 6, 7}:
            return position == 9
        elif attn_1_1_output in {8, 10}:
            return position == 11
        elif attn_1_1_output in {11}:
            return position == 18
        elif attn_1_1_output in {13}:
            return position == 14
        elif attn_1_1_output in {14}:
            return position == 8
        elif attn_1_1_output in {16}:
            return position == 19
        elif attn_1_1_output in {17, 19}:
            return position == 10
        elif attn_1_1_output in {18}:
            return position == 17

    num_attn_2_1_pattern = select(positions, attn_1_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_2_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_0_output, num_mlp_0_0_output):
        if attn_1_0_output in {0, 1, 2, 4, 10, 11, 12, 14, 16, 18, 19}:
            return num_mlp_0_0_output == 9
        elif attn_1_0_output in {3, 8, 9, 15, 17}:
            return num_mlp_0_0_output == 14
        elif attn_1_0_output in {5}:
            return num_mlp_0_0_output == 2
        elif attn_1_0_output in {6}:
            return num_mlp_0_0_output == 11
        elif attn_1_0_output in {13, 7}:
            return num_mlp_0_0_output == 13

    num_attn_2_2_pattern = select(
        num_mlp_0_0_outputs, attn_1_0_outputs, num_predicate_2_2
    )
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_0_0_output, position):
        if attn_0_0_output in {"<s>", "0", "1", "5", "3", "2", "4"}:
            return position == 0

    num_attn_2_3_pattern = select(positions, attn_0_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_0_output, token):
        key = (attn_2_0_output, token)
        return 3

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_0_outputs, tokens)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(mlp_0_1_output, attn_2_2_output):
        key = (mlp_0_1_output, attn_2_2_output)
        return 11

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(mlp_0_1_outputs, attn_2_2_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_2_output, num_attn_2_0_output):
        key = (num_attn_2_2_output, num_attn_2_0_output)
        return 17

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_2_outputs, num_attn_2_0_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_2_output, num_attn_2_1_output):
        key = (num_attn_1_2_output, num_attn_2_1_output)
        return 13

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_2_1_outputs)
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
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
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


print(run(["<s>", "5", "0", "3", "2", "3", "0", "2", "1", "3"]))

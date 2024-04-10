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
        "output/length/rasp/reverse/trainlength30/s4/reverse_weights.csv",
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
        if q_position in {0, 29}:
            return k_position == 29
        elif q_position in {32, 1, 3}:
            return k_position == 26
        elif q_position in {2}:
            return k_position == 16
        elif q_position in {4}:
            return k_position == 23
        elif q_position in {5}:
            return k_position == 19
        elif q_position in {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 23, 24, 28}:
            return k_position == 1
        elif q_position in {17, 20, 21, 22, 27}:
            return k_position == 2
        elif q_position in {26, 19}:
            return k_position == 3
        elif q_position in {25}:
            return k_position == 4
        elif q_position in {30}:
            return k_position == 39
        elif q_position in {35, 31}:
            return k_position == 36
        elif q_position in {33}:
            return k_position == 31
        elif q_position in {34}:
            return k_position == 24
        elif q_position in {36}:
            return k_position == 34
        elif q_position in {37}:
            return k_position == 33
        elif q_position in {38}:
            return k_position == 35
        elif q_position in {39}:
            return k_position == 30

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 26, 27, 28, 29}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 10
        elif q_position in {2}:
            return k_position == 8
        elif q_position in {33, 3}:
            return k_position == 19
        elif q_position in {4}:
            return k_position == 21
        elif q_position in {5, 7}:
            return k_position == 22
        elif q_position in {6}:
            return k_position == 17
        elif q_position in {8, 10, 19, 23, 24}:
            return k_position == 4
        elif q_position in {9}:
            return k_position == 16
        elif q_position in {11}:
            return k_position == 18
        elif q_position in {12, 15}:
            return k_position == 3
        elif q_position in {13}:
            return k_position == 15
        elif q_position in {18, 21, 14}:
            return k_position == 6
        elif q_position in {16, 17, 20, 22}:
            return k_position == 5
        elif q_position in {25}:
            return k_position == 2
        elif q_position in {38, 30}:
            return k_position == 33
        elif q_position in {37, 31}:
            return k_position == 39
        elif q_position in {32, 35, 36}:
            return k_position == 38
        elif q_position in {34}:
            return k_position == 36
        elif q_position in {39}:
            return k_position == 34

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 10
        elif q_position in {1}:
            return k_position == 28
        elif q_position in {2}:
            return k_position == 27
        elif q_position in {3, 6}:
            return k_position == 21
        elif q_position in {4, 5, 25, 26, 27, 28}:
            return k_position == 1
        elif q_position in {9, 7}:
            return k_position == 19
        elif q_position in {8}:
            return k_position == 17
        elif q_position in {10}:
            return k_position == 18
        elif q_position in {16, 17, 11}:
            return k_position == 6
        elif q_position in {12, 13}:
            return k_position == 4
        elif q_position in {14, 18, 19, 20, 22}:
            return k_position == 7
        elif q_position in {15}:
            return k_position == 5
        elif q_position in {21}:
            return k_position == 8
        elif q_position in {24, 23}:
            return k_position == 2
        elif q_position in {29}:
            return k_position == 0
        elif q_position in {30}:
            return k_position == 3
        elif q_position in {31}:
            return k_position == 16
        elif q_position in {32}:
            return k_position == 23
        elif q_position in {33}:
            return k_position == 24
        elif q_position in {34, 38}:
            return k_position == 39
        elif q_position in {35}:
            return k_position == 26
        elif q_position in {36}:
            return k_position == 38
        elif q_position in {37}:
            return k_position == 13
        elif q_position in {39}:
            return k_position == 29

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 37, 39}:
            return k_position == 19
        elif q_position in {1}:
            return k_position == 14
        elif q_position in {2}:
            return k_position == 12
        elif q_position in {3}:
            return k_position == 20
        elif q_position in {4}:
            return k_position == 25
        elif q_position in {5}:
            return k_position == 18
        elif q_position in {16, 10, 6}:
            return k_position == 9
        elif q_position in {7, 8, 13, 24, 25}:
            return k_position == 3
        elif q_position in {9, 20, 22}:
            return k_position == 6
        elif q_position in {11, 14, 15}:
            return k_position == 4
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {17}:
            return k_position == 7
        elif q_position in {18, 19, 21, 23}:
            return k_position == 5
        elif q_position in {26, 27}:
            return k_position == 2
        elif q_position in {28}:
            return k_position == 1
        elif q_position in {29}:
            return k_position == 0
        elif q_position in {30}:
            return k_position == 16
        elif q_position in {34, 31}:
            return k_position == 34
        elif q_position in {32}:
            return k_position == 30
        elif q_position in {33}:
            return k_position == 32
        elif q_position in {35}:
            return k_position == 13
        elif q_position in {36}:
            return k_position == 28
        elif q_position in {38}:
            return k_position == 26

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_token, k_token):
        if q_token in {"0", "2"}:
            return k_token == "4"
        elif q_token in {"3", "1"}:
            return k_token == "2"
        elif q_token in {"</s>", "4", "<s>"}:
            return k_token == "0"

    attn_0_4_pattern = select_closest(tokens, tokens, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0}:
            return k_position == 29
        elif q_position in {1, 2}:
            return k_position == 24
        elif q_position in {3}:
            return k_position == 23
        elif q_position in {4, 36}:
            return k_position == 2
        elif q_position in {5, 38}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {30, 7}:
            return k_position == 5
        elif q_position in {8, 10}:
            return k_position == 6
        elif q_position in {9, 11}:
            return k_position == 7
        elif q_position in {12, 14}:
            return k_position == 10
        elif q_position in {33, 13}:
            return k_position == 11
        elif q_position in {17, 37, 15}:
            return k_position == 13
        elif q_position in {16}:
            return k_position == 12
        elif q_position in {18}:
            return k_position == 14
        elif q_position in {19}:
            return k_position == 15
        elif q_position in {20, 22}:
            return k_position == 18
        elif q_position in {21, 23}:
            return k_position == 19
        elif q_position in {24, 26, 31}:
            return k_position == 22
        elif q_position in {25}:
            return k_position == 21
        elif q_position in {27}:
            return k_position == 25
        elif q_position in {28, 39}:
            return k_position == 26
        elif q_position in {29}:
            return k_position == 27
        elif q_position in {32}:
            return k_position == 36
        elif q_position in {34}:
            return k_position == 33
        elif q_position in {35}:
            return k_position == 37

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 26, 28, 29}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 20
        elif q_position in {2}:
            return k_position == 9
        elif q_position in {3}:
            return k_position == 8
        elif q_position in {4}:
            return k_position == 24
        elif q_position in {5}:
            return k_position == 10
        elif q_position in {6}:
            return k_position == 16
        elif q_position in {7}:
            return k_position == 12
        elif q_position in {8}:
            return k_position == 13
        elif q_position in {9}:
            return k_position == 15
        elif q_position in {10, 11, 12, 13, 14, 15, 16, 19, 27}:
            return k_position == 2
        elif q_position in {17, 18, 21, 22, 25}:
            return k_position == 3
        elif q_position in {20}:
            return k_position == 4
        elif q_position in {23}:
            return k_position == 6
        elif q_position in {24}:
            return k_position == 5
        elif q_position in {33, 30}:
            return k_position == 34
        elif q_position in {31}:
            return k_position == 36
        elif q_position in {32}:
            return k_position == 33
        elif q_position in {34, 38}:
            return k_position == 37
        elif q_position in {35}:
            return k_position == 39
        elif q_position in {36}:
            return k_position == 30
        elif q_position in {37, 39}:
            return k_position == 29

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0, 10, 11, 13, 24}:
            return k_position == 5
        elif q_position in {1, 5}:
            return k_position == 24
        elif q_position in {2, 4}:
            return k_position == 20
        elif q_position in {33, 3}:
            return k_position == 25
        elif q_position in {6}:
            return k_position == 23
        elif q_position in {8, 7}:
            return k_position == 21
        elif q_position in {9, 17, 18, 21, 22}:
            return k_position == 4
        elif q_position in {19, 12, 15}:
            return k_position == 6
        elif q_position in {14, 16, 20, 23, 26}:
            return k_position == 3
        elif q_position in {25, 27}:
            return k_position == 2
        elif q_position in {28, 29}:
            return k_position == 1
        elif q_position in {30}:
            return k_position == 35
        elif q_position in {31}:
            return k_position == 19
        elif q_position in {32, 38}:
            return k_position == 22
        elif q_position in {34}:
            return k_position == 38
        elif q_position in {35}:
            return k_position == 33
        elif q_position in {36}:
            return k_position == 34
        elif q_position in {37}:
            return k_position == 36
        elif q_position in {39}:
            return k_position == 37

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"</s>", "0", "1"}:
            return position == 38
        elif token in {"2"}:
            return position == 34
        elif token in {"3"}:
            return position == 39
        elif token in {"4"}:
            return position == 30
        elif token in {"<s>"}:
            return position == 18

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {"3", "0"}:
            return position == 31
        elif token in {"1"}:
            return position == 32
        elif token in {"2"}:
            return position == 37
        elif token in {"4"}:
            return position == 8
        elif token in {"</s>"}:
            return position == 13
        elif token in {"<s>"}:
            return position == 10

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0, 32, 35, 18, 21, 28}:
            return k_position == 34
        elif q_position in {1, 4, 9}:
            return k_position == 33
        elif q_position in {2, 37, 5, 7, 29}:
            return k_position == 39
        elif q_position in {3, 36, 22, 24, 31}:
            return k_position == 35
        elif q_position in {6, 39}:
            return k_position == 30
        elif q_position in {8, 11, 13}:
            return k_position == 37
        elif q_position in {25, 10, 34}:
            return k_position == 36
        elif q_position in {33, 12, 15}:
            return k_position == 32
        elif q_position in {14}:
            return k_position == 21
        elif q_position in {16}:
            return k_position == 2
        elif q_position in {17}:
            return k_position == 20
        elif q_position in {19, 23}:
            return k_position == 31
        elif q_position in {20}:
            return k_position == 25
        elif q_position in {26}:
            return k_position == 22
        elif q_position in {27}:
            return k_position == 38
        elif q_position in {30}:
            return k_position == 29
        elif q_position in {38}:
            return k_position == 28

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"0"}:
            return position == 35
        elif token in {"1"}:
            return position == 8
        elif token in {"2"}:
            return position == 36
        elif token in {"3"}:
            return position == 38
        elif token in {"4", "<s>"}:
            return position == 31
        elif token in {"</s>"}:
            return position == 10

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(token, position):
        if token in {"0"}:
            return position == 8
        elif token in {"3", "1"}:
            return position == 30
        elif token in {"2"}:
            return position == 33
        elif token in {"4"}:
            return position == 35
        elif token in {"</s>"}:
            return position == 13
        elif token in {"<s>"}:
            return position == 12

    num_attn_0_4_pattern = select(positions, tokens, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(token, position):
        if token in {"0"}:
            return position == 32
        elif token in {"1", "<s>"}:
            return position == 31
        elif token in {"2"}:
            return position == 33
        elif token in {"3"}:
            return position == 30
        elif token in {"4"}:
            return position == 39
        elif token in {"</s>"}:
            return position == 35

    num_attn_0_5_pattern = select(positions, tokens, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(token, position):
        if token in {"4", "0"}:
            return position == 34
        elif token in {"1"}:
            return position == 39
        elif token in {"2"}:
            return position == 36
        elif token in {"3"}:
            return position == 8
        elif token in {"</s>"}:
            return position == 13
        elif token in {"<s>"}:
            return position == 10

    num_attn_0_6_pattern = select(positions, tokens, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(q_token, k_token):
        if q_token in {"3", "0", "4", "2", "</s>", "1", "<s>"}:
            return k_token == ""

    num_attn_0_7_pattern = select(tokens, tokens, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_6_output, attn_0_1_output):
        key = (attn_0_6_output, attn_0_1_output)
        if key in {
            ("0", "4"),
            ("0", "<s>"),
            ("1", "4"),
            ("1", "<s>"),
            ("2", "4"),
            ("2", "<s>"),
            ("4", "0"),
            ("4", "1"),
            ("4", "2"),
            ("4", "4"),
            ("4", "</s>"),
            ("4", "<s>"),
            ("</s>", "4"),
            ("</s>", "<s>"),
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "2"),
            ("<s>", "4"),
            ("<s>", "</s>"),
            ("<s>", "<s>"),
        }:
            return 13
        return 2

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_1_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_6_output, attn_0_0_output):
        key = (attn_0_6_output, attn_0_0_output)
        return 3

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_0_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_7_output, num_attn_0_0_output):
        key = (num_attn_0_7_output, num_attn_0_0_output)
        return 2

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_7_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_5_output, num_attn_0_0_output):
        key = (num_attn_0_5_output, num_attn_0_0_output)
        return 6

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_5_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_0_output, token):
        if attn_0_0_output in {"0", "1"}:
            return token == ""
        elif attn_0_0_output in {"3", "2"}:
            return token == "<s>"
        elif attn_0_0_output in {"4"}:
            return token == "0"
        elif attn_0_0_output in {"</s>"}:
            return token == "4"
        elif attn_0_0_output in {"<s>"}:
            return token == "</s>"

    attn_1_0_pattern = select_closest(tokens, attn_0_0_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_4_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_0_output, token):
        if attn_0_0_output in {"0"}:
            return token == "0"
        elif attn_0_0_output in {"3", "1", "2"}:
            return token == "4"
        elif attn_0_0_output in {"4"}:
            return token == "3"
        elif attn_0_0_output in {"</s>"}:
            return token == "2"
        elif attn_0_0_output in {"<s>"}:
            return token == "1"

    attn_1_1_pattern = select_closest(tokens, attn_0_0_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_0_output, token):
        if attn_0_0_output in {"4", "3", "0", "2"}:
            return token == ""
        elif attn_0_0_output in {"1"}:
            return token == "1"
        elif attn_0_0_output in {"</s>"}:
            return token == "</s>"
        elif attn_0_0_output in {"<s>"}:
            return token == "3"

    attn_1_2_pattern = select_closest(tokens, attn_0_0_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_0_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_7_output, token):
        if attn_0_7_output in {"0", "1", "<s>", "2"}:
            return token == ""
        elif attn_0_7_output in {"</s>", "4", "3"}:
            return token == "4"

    attn_1_3_pattern = select_closest(tokens, attn_0_7_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(q_position, k_position):
        if q_position in {0, 18, 26}:
            return k_position == 2
        elif q_position in {1, 9}:
            return k_position == 11
        elif q_position in {2}:
            return k_position == 27
        elif q_position in {3, 6}:
            return k_position == 22
        elif q_position in {4}:
            return k_position == 16
        elif q_position in {35, 5}:
            return k_position == 15
        elif q_position in {7}:
            return k_position == 9
        elif q_position in {8, 13, 14}:
            return k_position == 0
        elif q_position in {10}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 13
        elif q_position in {12}:
            return k_position == 14
        elif q_position in {15}:
            return k_position == 29
        elif q_position in {16}:
            return k_position == 4
        elif q_position in {17, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29}:
            return k_position == 1
        elif q_position in {30}:
            return k_position == 25
        elif q_position in {38, 31}:
            return k_position == 38
        elif q_position in {32}:
            return k_position == 39
        elif q_position in {33}:
            return k_position == 37
        elif q_position in {34}:
            return k_position == 24
        elif q_position in {36}:
            return k_position == 5
        elif q_position in {37}:
            return k_position == 31
        elif q_position in {39}:
            return k_position == 36

    attn_1_4_pattern = select_closest(positions, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, tokens)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(attn_0_6_output, position):
        if attn_0_6_output in {"0"}:
            return position == 20
        elif attn_0_6_output in {"1"}:
            return position == 26
        elif attn_0_6_output in {"2"}:
            return position == 8
        elif attn_0_6_output in {"3"}:
            return position == 25
        elif attn_0_6_output in {"4"}:
            return position == 29
        elif attn_0_6_output in {"</s>"}:
            return position == 5
        elif attn_0_6_output in {"<s>"}:
            return position == 36

    attn_1_5_pattern = select_closest(positions, attn_0_6_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, tokens)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(position, token):
        if position in {
            0,
            1,
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
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
        }:
            return token == "<s>"
        elif position in {2, 4, 39}:
            return token == "</s>"
        elif position in {32, 34, 3, 5, 37, 30}:
            return token == "3"
        elif position in {35, 36, 38, 7}:
            return token == ""
        elif position in {29}:
            return token == "1"
        elif position in {31}:
            return token == "0"
        elif position in {33}:
            return token == "2"

    attn_1_6_pattern = select_closest(tokens, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_5_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_0_output, token):
        if attn_0_0_output in {"</s>", "0", "1", "<s>"}:
            return token == "3"
        elif attn_0_0_output in {"2"}:
            return token == "2"
        elif attn_0_0_output in {"3"}:
            return token == "4"
        elif attn_0_0_output in {"4"}:
            return token == "0"

    attn_1_7_pattern = select_closest(tokens, attn_0_0_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, tokens)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(mlp_0_1_output, position):
        if mlp_0_1_output in {0, 34}:
            return position == 5
        elif mlp_0_1_output in {1}:
            return position == 14
        elif mlp_0_1_output in {17, 2, 33}:
            return position == 27
        elif mlp_0_1_output in {3}:
            return position == 30
        elif mlp_0_1_output in {4}:
            return position == 33
        elif mlp_0_1_output in {5, 14, 30}:
            return position == 21
        elif mlp_0_1_output in {6}:
            return position == 6
        elif mlp_0_1_output in {39, 8, 9, 7, 31}:
            return position == 19
        elif mlp_0_1_output in {10, 12}:
            return position == 13
        elif mlp_0_1_output in {11, 21, 23}:
            return position == 23
        elif mlp_0_1_output in {13}:
            return position == 10
        elif mlp_0_1_output in {15}:
            return position == 22
        elif mlp_0_1_output in {16}:
            return position == 34
        elif mlp_0_1_output in {18, 35}:
            return position == 4
        elif mlp_0_1_output in {19}:
            return position == 11
        elif mlp_0_1_output in {32, 20}:
            return position == 2
        elif mlp_0_1_output in {22}:
            return position == 8
        elif mlp_0_1_output in {24}:
            return position == 1
        elif mlp_0_1_output in {25}:
            return position == 7
        elif mlp_0_1_output in {26}:
            return position == 25
        elif mlp_0_1_output in {27}:
            return position == 29
        elif mlp_0_1_output in {28}:
            return position == 18
        elif mlp_0_1_output in {29}:
            return position == 31
        elif mlp_0_1_output in {36}:
            return position == 17
        elif mlp_0_1_output in {37}:
            return position == 3
        elif mlp_0_1_output in {38}:
            return position == 15

    num_attn_1_0_pattern = select(positions, mlp_0_1_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_4_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, num_mlp_0_0_output):
        if position in {0, 8}:
            return num_mlp_0_0_output == 21
        elif position in {1}:
            return num_mlp_0_0_output == 10
        elif position in {9, 2}:
            return num_mlp_0_0_output == 20
        elif position in {3}:
            return num_mlp_0_0_output == 12
        elif position in {4, 15}:
            return num_mlp_0_0_output == 28
        elif position in {12, 5}:
            return num_mlp_0_0_output == 5
        elif position in {6, 23}:
            return num_mlp_0_0_output == 1
        elif position in {22, 7}:
            return num_mlp_0_0_output == 37
        elif position in {10}:
            return num_mlp_0_0_output == 8
        elif position in {11, 28}:
            return num_mlp_0_0_output == 34
        elif position in {13}:
            return num_mlp_0_0_output == 18
        elif position in {14}:
            return num_mlp_0_0_output == 30
        elif position in {16, 26, 37}:
            return num_mlp_0_0_output == 7
        elif position in {17}:
            return num_mlp_0_0_output == 36
        elif position in {18}:
            return num_mlp_0_0_output == 16
        elif position in {34, 19, 36}:
            return num_mlp_0_0_output == 2
        elif position in {20}:
            return num_mlp_0_0_output == 25
        elif position in {21}:
            return num_mlp_0_0_output == 22
        elif position in {24}:
            return num_mlp_0_0_output == 14
        elif position in {25}:
            return num_mlp_0_0_output == 39
        elif position in {33, 27}:
            return num_mlp_0_0_output == 33
        elif position in {29}:
            return num_mlp_0_0_output == 35
        elif position in {30}:
            return num_mlp_0_0_output == 3
        elif position in {31}:
            return num_mlp_0_0_output == 11
        elif position in {32, 35}:
            return num_mlp_0_0_output == 4
        elif position in {38}:
            return num_mlp_0_0_output == 0
        elif position in {39}:
            return num_mlp_0_0_output == 27

    num_attn_1_1_pattern = select(num_mlp_0_0_outputs, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_2_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(q_attn_0_0_output, k_attn_0_0_output):
        if q_attn_0_0_output in {"0"}:
            return k_attn_0_0_output == "<pad>"
        elif q_attn_0_0_output in {"3", "1", "2"}:
            return k_attn_0_0_output == ""
        elif q_attn_0_0_output in {"4", "<s>"}:
            return k_attn_0_0_output == "4"
        elif q_attn_0_0_output in {"</s>"}:
            return k_attn_0_0_output == "</s>"

    num_attn_1_2_pattern = select(attn_0_0_outputs, attn_0_0_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(num_mlp_0_1_output, attn_0_7_output):
        if num_mlp_0_1_output in {
            0,
            1,
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
            18,
            19,
            20,
            21,
            23,
            24,
            25,
            26,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        }:
            return attn_0_7_output == ""
        elif num_mlp_0_1_output in {2, 14, 22}:
            return attn_0_7_output == "<pad>"
        elif num_mlp_0_1_output in {27}:
            return attn_0_7_output == "</s>"

    num_attn_1_3_pattern = select(
        attn_0_7_outputs, num_mlp_0_1_outputs, num_predicate_1_3
    )
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_5_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(q_attn_0_0_output, k_attn_0_0_output):
        if q_attn_0_0_output in {"3", "0", "4", "2", "</s>", "<s>"}:
            return k_attn_0_0_output == ""
        elif q_attn_0_0_output in {"1"}:
            return k_attn_0_0_output == "1"

    num_attn_1_4_pattern = select(attn_0_0_outputs, attn_0_0_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_2_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(attn_0_6_output, attn_0_3_output):
        if attn_0_6_output in {"3", "0", "4", "</s>", "1", "<s>"}:
            return attn_0_3_output == ""
        elif attn_0_6_output in {"2"}:
            return attn_0_3_output == "<pad>"

    num_attn_1_5_pattern = select(attn_0_3_outputs, attn_0_6_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_2_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(attn_0_7_output, attn_0_0_output):
        if attn_0_7_output in {"3", "0", "4", "2", "</s>", "1"}:
            return attn_0_0_output == ""
        elif attn_0_7_output in {"<s>"}:
            return attn_0_0_output == "3"

    num_attn_1_6_pattern = select(attn_0_0_outputs, attn_0_7_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_1_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(attn_0_0_output, attn_0_1_output):
        if attn_0_0_output in {"0"}:
            return attn_0_1_output == "0"
        elif attn_0_0_output in {"4", "1", "2"}:
            return attn_0_1_output == "<pad>"
        elif attn_0_0_output in {"</s>", "3", "<s>"}:
            return attn_0_1_output == ""

    num_attn_1_7_pattern = select(attn_0_1_outputs, attn_0_0_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_6_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_3_output):
        key = attn_0_3_output
        if key in {"", "0", "2", "4", "</s>", "<pad>", "<s>"}:
            return 15
        return 20

    mlp_1_0_outputs = [mlp_1_0(k0) for k0 in attn_0_3_outputs]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_7_output, attn_1_0_output):
        key = (attn_1_7_output, attn_1_0_output)
        return 4

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_7_outputs, attn_1_0_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_6_output, num_attn_1_1_output):
        key = (num_attn_1_6_output, num_attn_1_1_output)
        return 6

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_5_output, num_attn_0_5_output):
        key = (num_attn_1_5_output, num_attn_0_5_output)
        return 32

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_5_outputs, num_attn_0_5_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_0_output, token):
        if attn_0_0_output in {"4", "0"}:
            return token == ""
        elif attn_0_0_output in {"1"}:
            return token == "1"
        elif attn_0_0_output in {"2"}:
            return token == "2"
        elif attn_0_0_output in {"3"}:
            return token == "3"
        elif attn_0_0_output in {"</s>"}:
            return token == "</s>"
        elif attn_0_0_output in {"<s>"}:
            return token == "4"

    attn_2_0_pattern = select_closest(tokens, attn_0_0_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_0_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_1_4_output, token):
        if attn_1_4_output in {"</s>", "3", "0", "2"}:
            return token == ""
        elif attn_1_4_output in {"1"}:
            return token == "</s>"
        elif attn_1_4_output in {"4"}:
            return token == "3"
        elif attn_1_4_output in {"<s>"}:
            return token == "1"

    attn_2_1_pattern = select_closest(tokens, attn_1_4_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_7_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_6_output, token):
        if attn_0_6_output in {"0"}:
            return token == "<s>"
        elif attn_0_6_output in {"1", "<s>"}:
            return token == ""
        elif attn_0_6_output in {"3", "2"}:
            return token == "0"
        elif attn_0_6_output in {"4"}:
            return token == "</s>"
        elif attn_0_6_output in {"</s>"}:
            return token == "1"

    attn_2_2_pattern = select_closest(tokens, attn_0_6_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_0_output, num_mlp_0_0_output):
        if attn_0_0_output in {"0"}:
            return num_mlp_0_0_output == 4
        elif attn_0_0_output in {"1"}:
            return num_mlp_0_0_output == 3
        elif attn_0_0_output in {"2"}:
            return num_mlp_0_0_output == 25
        elif attn_0_0_output in {"3"}:
            return num_mlp_0_0_output == 7
        elif attn_0_0_output in {"4"}:
            return num_mlp_0_0_output == 6
        elif attn_0_0_output in {"</s>"}:
            return num_mlp_0_0_output == 27
        elif attn_0_0_output in {"<s>"}:
            return num_mlp_0_0_output == 13

    attn_2_3_pattern = select_closest(
        num_mlp_0_0_outputs, attn_0_0_outputs, predicate_2_3
    )
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_0_output, token):
        if attn_0_0_output in {"4", "3", "0"}:
            return token == "<s>"
        elif attn_0_0_output in {"1", "2"}:
            return token == ""
        elif attn_0_0_output in {"</s>", "<s>"}:
            return token == "3"

    attn_2_4_pattern = select_closest(tokens, attn_0_0_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_1_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(num_mlp_0_0_output, position):
        if num_mlp_0_0_output in {0, 12}:
            return position == 9
        elif num_mlp_0_0_output in {1, 5, 17, 25}:
            return position == 12
        elif num_mlp_0_0_output in {2, 18, 6}:
            return position == 3
        elif num_mlp_0_0_output in {3}:
            return position == 16
        elif num_mlp_0_0_output in {4}:
            return position == 0
        elif num_mlp_0_0_output in {7}:
            return position == 18
        elif num_mlp_0_0_output in {8}:
            return position == 6
        elif num_mlp_0_0_output in {9}:
            return position == 2
        elif num_mlp_0_0_output in {10}:
            return position == 19
        elif num_mlp_0_0_output in {19, 27, 11, 20}:
            return position == 7
        elif num_mlp_0_0_output in {13}:
            return position == 20
        elif num_mlp_0_0_output in {14}:
            return position == 30
        elif num_mlp_0_0_output in {26, 15}:
            return position == 13
        elif num_mlp_0_0_output in {16, 22}:
            return position == 5
        elif num_mlp_0_0_output in {21}:
            return position == 1
        elif num_mlp_0_0_output in {38, 23}:
            return position == 14
        elif num_mlp_0_0_output in {24, 33}:
            return position == 24
        elif num_mlp_0_0_output in {28}:
            return position == 23
        elif num_mlp_0_0_output in {29}:
            return position == 26
        elif num_mlp_0_0_output in {30}:
            return position == 15
        elif num_mlp_0_0_output in {31}:
            return position == 11
        elif num_mlp_0_0_output in {32}:
            return position == 37
        elif num_mlp_0_0_output in {34, 35}:
            return position == 10
        elif num_mlp_0_0_output in {36}:
            return position == 31
        elif num_mlp_0_0_output in {37}:
            return position == 22
        elif num_mlp_0_0_output in {39}:
            return position == 21

    attn_2_5_pattern = select_closest(positions, num_mlp_0_0_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_7_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_0_0_output, attn_0_5_output):
        if attn_0_0_output in {"0"}:
            return attn_0_5_output == "2"
        elif attn_0_0_output in {"4", "3", "1", "<s>"}:
            return attn_0_5_output == ""
        elif attn_0_0_output in {"2"}:
            return attn_0_5_output == "4"
        elif attn_0_0_output in {"</s>"}:
            return attn_0_5_output == "3"

    attn_2_6_pattern = select_closest(attn_0_5_outputs, attn_0_0_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_3_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_6_output, position):
        if attn_0_6_output in {"0"}:
            return position == 7
        elif attn_0_6_output in {"1"}:
            return position == 1
        elif attn_0_6_output in {"2"}:
            return position == 5
        elif attn_0_6_output in {"3"}:
            return position == 10
        elif attn_0_6_output in {"4"}:
            return position == 9
        elif attn_0_6_output in {"</s>"}:
            return position == 23
        elif attn_0_6_output in {"<s>"}:
            return position == 12

    attn_2_7_pattern = select_closest(positions, attn_0_6_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_2_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_4_output, attn_1_7_output):
        if attn_1_4_output in {"3", "0", "2", "</s>", "1", "<s>"}:
            return attn_1_7_output == ""
        elif attn_1_4_output in {"4"}:
            return attn_1_7_output == "4"

    num_attn_2_0_pattern = select(attn_1_7_outputs, attn_1_4_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_0_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_3_output, attn_0_0_output):
        if attn_1_3_output in {"3", "0", "4", "2", "</s>", "1", "<s>"}:
            return attn_0_0_output == ""

    num_attn_2_1_pattern = select(attn_0_0_outputs, attn_1_3_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_5_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(q_attn_1_6_output, k_attn_1_6_output):
        if q_attn_1_6_output in {"0", "4", "2", "</s>", "1", "<s>"}:
            return k_attn_1_6_output == ""
        elif q_attn_1_6_output in {"3"}:
            return k_attn_1_6_output == "3"

    num_attn_2_2_pattern = select(attn_1_6_outputs, attn_1_6_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_6_output, attn_0_1_output):
        if attn_1_6_output in {"0"}:
            return attn_0_1_output == "0"
        elif attn_1_6_output in {"3", "4", "2", "</s>", "1", "<s>"}:
            return attn_0_1_output == ""

    num_attn_2_3_pattern = select(attn_0_1_outputs, attn_1_6_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_1_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_0_0_output, attn_1_4_output):
        if attn_0_0_output in {"3", "0", "4", "</s>", "1"}:
            return attn_1_4_output == ""
        elif attn_0_0_output in {"<s>", "2"}:
            return attn_1_4_output == "2"

    num_attn_2_4_pattern = select(attn_1_4_outputs, attn_0_0_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_1_1_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_1_1_output, attn_0_1_output):
        if attn_1_1_output in {"3", "0", "4", "2", "</s>", "1", "<s>"}:
            return attn_0_1_output == ""

    num_attn_2_5_pattern = select(attn_0_1_outputs, attn_1_1_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_1_5_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_0_6_output, attn_1_4_output):
        if attn_0_6_output in {"3", "0", "2", "</s>", "1", "<s>"}:
            return attn_1_4_output == ""
        elif attn_0_6_output in {"4"}:
            return attn_1_4_output == "4"

    num_attn_2_6_pattern = select(attn_1_4_outputs, attn_0_6_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_0_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_0_0_output, attn_1_4_output):
        if attn_0_0_output in {"0", "4", "2", "</s>", "1", "<s>"}:
            return attn_1_4_output == ""
        elif attn_0_0_output in {"3"}:
            return attn_1_4_output == "3"

    num_attn_2_7_pattern = select(attn_1_4_outputs, attn_0_0_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_2_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_5_output, attn_0_4_output):
        key = (attn_2_5_output, attn_0_4_output)
        return 5

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_5_outputs, attn_0_4_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_1_4_output, attn_0_6_output):
        key = (attn_1_4_output, attn_0_6_output)
        if key in {
            ("0", "0"),
            ("0", "2"),
            ("1", "0"),
            ("1", "2"),
            ("2", "0"),
            ("2", "2"),
            ("3", "0"),
            ("3", "2"),
            ("4", "0"),
            ("4", "2"),
            ("</s>", "0"),
            ("</s>", "1"),
            ("</s>", "2"),
            ("</s>", "3"),
            ("</s>", "4"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
            ("<s>", "0"),
            ("<s>", "2"),
        }:
            return 3
        elif key in {
            ("0", "4"),
            ("0", "<s>"),
            ("1", "4"),
            ("1", "<s>"),
            ("2", "4"),
            ("2", "<s>"),
            ("3", "4"),
            ("3", "<s>"),
            ("4", "4"),
            ("4", "<s>"),
            ("<s>", "4"),
            ("<s>", "<s>"),
        }:
            return 17
        elif key in {
            ("0", "3"),
            ("1", "3"),
            ("2", "3"),
            ("3", "3"),
            ("4", "3"),
            ("<s>", "3"),
        }:
            return 6
        return 11

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_1_4_outputs, attn_0_6_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_2_output, num_attn_1_5_output):
        key = (num_attn_2_2_output, num_attn_1_5_output)
        return 6

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_2_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_1_output, num_attn_1_0_output):
        key = (num_attn_2_1_output, num_attn_1_0_output)
        return 38

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_1_0_outputs)
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


print(
    run(
        [
            "<s>",
            "1",
            "0",
            "0",
            "2",
            "1",
            "2",
            "4",
            "1",
            "0",
            "4",
            "2",
            "4",
            "2",
            "4",
            "3",
            "0",
            "1",
            "0",
            "2",
            "0",
            "1",
            "2",
            "2",
            "0",
            "0",
            "3",
            "2",
            "</s>",
        ]
    )
)

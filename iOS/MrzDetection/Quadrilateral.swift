//
//  Quadrilatoral.swift
//  MrzDetection
//
//  Created by Giles Payne on 2021/10/10.
//

import Foundation
import opencv2

public class Quadrilateral {
    public let p1: Point2d;
    public let p2: Point2d;
    public let p3: Point2d;
    public let p4: Point2d;

    init(p1: Point2d, p2: Point2d, p3: Point2d, p4: Point2d) {
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
    }
}

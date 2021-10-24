//
//  MrzDetectionTests.swift
//  MrzDetectionTests
//
//  Created by Giles Payne on 2021/10/10.
//

import XCTest
@testable import MrzDetection
import opencv2

class MrzDetectionTests: XCTestCase {

    let data1: Mat = {
        return Imgcodecs.imread(filename: Bundle(for: MrzDetectionTests.self).path(forResource:"data1", ofType:"jpg", inDirectory:"data")!)
    }()

    let data6: Mat = {
        return Imgcodecs.imread(filename: Bundle(for: MrzDetectionTests.self).path(forResource:"data6", ofType:"png", inDirectory:"data")!)
    }()

    let data7: Mat = {
        return Imgcodecs.imread(filename: Bundle(for: MrzDetectionTests.self).path(forResource:"data7", ofType:"jpg", inDirectory:"data")!)
    }()

    func testData1() throws {
        let result = MrzDetector().getMrz(src: data1)
        print(result ?? "Mrz not found")
    }

    func testData6() throws {
        let result = MrzDetector().getMrz(src: data6)
        print(result ?? "Mrz not found")
    }

    func testData7() throws {
        let result = MrzDetector().getMrz(src: data7)
        print(result ?? "Mrz not found")
    }

}

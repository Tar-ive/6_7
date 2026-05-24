import AVFoundation
import Foundation

let args = CommandLine.arguments
guard args.count >= 3 else {
    exit(2)
}

let player = try AVAudioPlayer(contentsOf: URL(fileURLWithPath: args[1]))
player.numberOfLoops = -1
player.volume = Float(args[2]) ?? 0.8
player.prepareToPlay()
player.play()

DispatchQueue.global(qos: .userInitiated).async {
    while let line = readLine() {
        let parts = line.split(separator: " ")
        if parts.first == "volume", parts.count == 2, let volume = Float(parts[1]) {
            DispatchQueue.main.async {
                player.volume = volume
            }
        } else if parts.first == "stop" {
            DispatchQueue.main.async {
                player.stop()
                CFRunLoopStop(CFRunLoopGetMain())
            }
            break
        }
    }
}

RunLoop.main.run()

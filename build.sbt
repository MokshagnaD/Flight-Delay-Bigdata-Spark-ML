ThisBuild / scalaVersion     := "2.12.10"
ThisBuild / version          := "0.1.0"
ThisBuild / organization     := "com.spark.example"
ThisBuild / organizationName := "example"

val sparkVersion = "2.4.5"

lazy val scalaTest = "org.scalatest" %% "scalatest" % "3.1.1"
lazy val sparkCore = "org.apache.spark" %% "spark-core" % sparkVersion
lazy val sparkSql = "org.apache.spark" %% "spark-sql" % sparkVersion
lazy val sparkMllib = "org.apache.spark" %% "spark-mllib" % sparkVersion

libraryDependencies ++= Seq(scalaTest % Test, sparkCore, sparkSql, sparkMllib)

libraryDependencies += "com.github.mrpowers" %% "spark-fast-tests" % "0.21.3" % Test

fork in run := true
javaOptions in run ++= Seq("-Xmx6g", "-XX:+UseG1GC")
